# File formats

All files are JSON unless stated otherwise. Relative paths are resolved
against the directory of the file that contains them.

## Experiment configuration (`tinydeploy run`)

| Key | Type | Meaning |
|---|---|---|
| `metric` | `"accuracy"` or `"auc"` | Task metric; every profile must use it |
| `minimum_score` | number in [0, 1] | Minimum validation score |
| `target.id` | string | Must match each profile's `target` |
| `target.sram_bytes`, `target.flash_bytes` | number | Budgets |
| `target.firmware_reserve_bytes` | number, optional | Flash kept for the application; model + reserve must fit in flash |
| `target.max_latency_ms` | number | Latency budget |
| `target.current_ma`, `target.voltage_v` | number | Active current and supply voltage for `E = t x I x V` |
| `target.supported_bit_widths` | list, optional (default `[8]`) | Bit-widths the target runtime can execute |
| `candidates` | list, optional | Explicit candidates: `id`, `model`, `profile`, optional `command` (argv list) run first |
| `grid` | object, optional | Algorithm 1 grid, see below |
| `build` | object, optional | `command` (argv; `{model_header}` is replaced) and `binary` (firmware file it produces) |

### `grid`

| Key | Meaning |
|---|---|
| `base_model` | Model to compress |
| `bit_widths` | Default `[8, 4]` |
| `prune_ratios` | Default `[0, 0.3, 0.5, 0.7]` |
| `model_file`, `profile_file` | Templates for the files each candidate must produce |
| `compress`, `profile` | Argument-list templates run for every supported configuration |

Placeholders: `{base_model}`, `{bits}`, `{prune}`, `{prune_pct}`, `{workdir}`,
`{model}`, `{profile}`, `{target}`. Candidate ids are `int<bits>-p<percent>`,
e.g. `int8-p30`.

## Profile

```json
{
  "schema_version": 1,
  "target": "stm32l476",
  "metric": "accuracy",
  "score": 0.912,
  "peak_sram_bytes": 61440,
  "latency_ms": 90.0,
  "model_sha256": "<sha256 of the model file>",
  "provenance": {"kind": "profiler_estimate", "tool": "edge_impulse",
                 "sources": ["edge_impulse_report.json"]}
}
```

`provenance.kind` is one of:

- `profiler_estimate` - device-specific profiler output; must cite `sources`.
- `measured` - must also contain `provenance.protocol` with `warmup`,
  `repetitions`, `timer`, `compiler`, `flags`, `clock_hz`, and
  `latency_samples_ms` with one value per repetition whose mean equals
  `latency_ms` (within 1%). The standard deviation is added on import.
- `synthetic` - generated data; `tinydeploy run` refuses it unless
  `--allow-synthetic` is given, and the run record then sets `contains_synthetic`.

## Run directory (`tinydeploy run --output DIR`)

```
DIR/config.json          copy of the configuration
DIR/results.json         every candidate with rejection_reasons, skipped configurations,
                         feasible_ids, pareto_ids, selected, firmware hash, wall time
DIR/profiles/<id>.json   archived profiles
DIR/logs/<id>-<step>.json   compress / profile / worker command logs
DIR/logs/build.json      build log
DIR/candidates/<id>/     grid working directories
DIR/selected_model.bin, DIR/model_data.h, DIR/firmware.bin
```

## OTA fleet configuration (`tinydeploy simulate-ota`)

| Key | Meaning |
|---|---|
| `devices`, `rounds` | Fleet size and update rounds (rounds alternate new/old image) |
| `seed` | Random seed (losses use `seed`, injected faults `seed + 1`) |
| `packet_bytes`, `retry_limit`, `packet_loss` | Link model |
| `bitrate_bps`, `radio_power_mw` | Airtime and radio energy |
| `post_verify_fault_probability` | Probability that an image is corrupted after verification (self-test then fails and the device rolls back) |
| `flash_bytes`, `bootloader_bytes`, `metadata_bytes`, `scratch_bytes` | Flash layout; each A/B slot is `(flash - reserved) / 2` |

Outputs: `events.jsonl` (one line per update: mode, device, round, status,
wire and on-air bytes, fault flag, active image hash) and `summary.json`.

## Delta

`{"schema_version": 1, "base_sha256", "new_sha256", "new_size", "runs": [[offset, base64], ...]}`.
Applying it checks the base hash, run bounds and the reconstructed image hash.

## Benchmark CSV (`firmware/tinydeploy_benchmark.h`)

`sample_index,elapsed_ticks,timer_hz`, one row per timed inference,
consecutive indices from 0.
