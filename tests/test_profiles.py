import pytest

from tinydeploy import TinyDeployError
from tinydeploy._util import digest, read_json, write_json
from tinydeploy.profiles import from_benchmark_csv, from_edge_impulse, validate_profile


def test_measured_profile_needs_protocol_and_raw_samples(selection_config):
    base = selection_config.parent
    profile = read_json(base / "large.json")
    profile["provenance"] = {"kind": "measured", "sources": ["large.bin"]}
    with pytest.raises(TinyDeployError, match="protocol"):
        validate_profile(profile, base)
    profile["provenance"]["protocol"] = {"warmup": 10, "repetitions": 3, "timer": "DWT cycle counter",
                                         "compiler": "arm-none-eabi-gcc", "flags": "-O3",
                                         "clock_hz": 80_000_000}
    profile["latency_samples_ms"] = [19, 20, 21]
    assert validate_profile(profile, base)["latency_stdev_ms"] == 1
    profile["latency_ms"] = 100
    with pytest.raises(TinyDeployError, match="mean"):
        validate_profile(profile, base)


def test_edge_impulse_report_import(tmp_path):
    model = tmp_path / "model.tflite"
    model.write_bytes(b"model-bytes")
    # Shape of an Edge Impulse profiling response for one device.
    report = {"success": True, "model": {"profileInfo": {"int8": {
        "device": "nordic-nrf52840-dk", "timePerInferenceMs": 2, "isSupportedOnMcu": True,
        "memory": {"tflite": {"ram": 4520, "rom": 138928}, "eon": {"ram": 2472, "rom": 110376}}}}}}
    write_json(tmp_path / "ei.json", report)
    profile = from_edge_impulse(tmp_path / "ei.json", model, target="nrf52840",
                                metric="auc", score=0.95)
    assert profile["peak_sram_bytes"] == 2472 and profile["flash_bytes"] == 110376
    assert profile["latency_ms"] == 2 and profile["model_sha256"] == digest(b"model-bytes")
    assert profile["provenance"]["kind"] == "profiler_estimate"
    validate_profile(profile, tmp_path)
    tflm = from_edge_impulse(tmp_path / "ei.json", model, target="nrf52840", metric="auc",
                             score=0.95, engine="tflite")
    assert tflm["peak_sram_bytes"] == 4520


def test_edge_impulse_rejects_other_json(tmp_path):
    (tmp_path / "m.bin").write_bytes(b"m")
    write_json(tmp_path / "x.json", {"unrelated": 1})
    with pytest.raises(TinyDeployError, match="Edge Impulse"):
        from_edge_impulse(tmp_path / "x.json", tmp_path / "m.bin", target="t", metric="accuracy", score=0.5)


def test_benchmark_csv_import(tmp_path):
    (tmp_path / "m.bin").write_bytes(b"m")
    rows = ["sample_index,elapsed_ticks,timer_hz"] + [f"{i},{7_200_000 + 80_000 * (i % 3)},80000000"
                                                      for i in range(30)]
    (tmp_path / "timing.csv").write_text("\n".join(rows) + "\n")
    protocol = {"warmup": 5, "repetitions": 30, "timer": "DWT cycle counter",
                "compiler": "arm-none-eabi-gcc 13.2", "flags": "-O3", "clock_hz": 80_000_000}
    profile = from_benchmark_csv(tmp_path / "timing.csv", tmp_path / "m.bin", target="stm32l476",
                                 metric="accuracy", score=0.9, peak_sram_bytes=60_000, protocol=protocol)
    assert profile["latency_ms"] == pytest.approx(91.0)
    assert len(profile["latency_samples_ms"]) == 30
    assert validate_profile(profile, tmp_path)["latency_stdev_ms"] > 0
    protocol["repetitions"] = 31
    with pytest.raises(TinyDeployError, match="repetitions"):
        from_benchmark_csv(tmp_path / "timing.csv", tmp_path / "m.bin", target="stm32l476",
                           metric="accuracy", score=0.9, peak_sram_bytes=1, protocol=protocol)
