"""A/B OTA update model and seeded fleet simulation (host-side).

Each simulated device keeps two firmware slots. An update is downloaded into
the inactive slot (as a full image or a delta), its SHA-256 is verified, a
post-boot self-test runs, and the device commits the new image or rolls back
to the previous one. Radio energy is transmitted bits / bitrate x radio power.

This is a simulation of the update logic, not MCU bootloader firmware: power-loss
atomicity, radio headers/ACKs and flash/CPU energy are outside its scope and are
listed under ``assumptions`` in every summary.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ._util import TinyDeployError, check_int, check_number, digest, log, write_json
from .delta import apply_delta, make_delta, wire

ASSUMPTIONS = [
    "Rounds alternate between the supplied new and old images",
    "No radio headers, ACKs or duty-cycle limits",
    "Constant radio power; flash, CPU and boot energy excluded",
    "Post-boot self-test is a host hash check standing in for the golden inference test",
    "Post-verification fault probability is an input parameter",
]
INT_KEYS = ("devices", "rounds", "packet_bytes", "retry_limit", "seed", "flash_bytes",
            "bootloader_bytes", "metadata_bytes", "scratch_bytes")
LAYOUT_KEYS = ("flash_bytes", "bootloader_bytes", "metadata_bytes", "scratch_bytes")


@dataclass
class Device:
    """Flash layout of one device: two equal slots after reserved regions."""

    image: bytes
    flash_bytes: int
    bootloader_bytes: int
    metadata_bytes: int
    scratch_bytes: int

    @property
    def slot_bytes(self) -> int:
        reserved = self.bootloader_bytes + self.metadata_bytes + self.scratch_bytes
        return (self.flash_bytes - reserved) // 2

    def update(self, payload: Any, expected_hash: str, self_test: Callable[[bytes], bool],
               delta: bool = False, corrupt_after_verify: bool = False) -> str:
        """Install ``payload`` into the inactive slot.

        Returns ``committed``, ``rolled_back`` (self-test failed; previous image
        kept) or ``integrity_rejected`` (hash mismatch; nothing installed).
        """
        prior = self.image
        candidate = apply_delta(prior, payload) if delta else payload
        if len(prior) > self.slot_bytes or len(candidate) > self.slot_bytes:
            raise TinyDeployError("image does not fit the A/B flash layout")
        if digest(candidate) != expected_hash:
            return "integrity_rejected"
        if corrupt_after_verify and candidate:
            candidate = bytes([candidate[0] ^ 1]) + candidate[1:]
        if not self_test(candidate):
            self.image = prior
            return "rolled_back"
        self.image = candidate
        return "committed"


def transfer(size: int, rng: random.Random, loss: float, packet_bytes: int,
             retry_limit: int) -> tuple[int, bool]:
    """Send ``size`` bytes in fixed packets; a lost packet costs full airtime.

    Returns (bytes put on air, completed). A packet lost more than
    ``retry_limit`` times aborts the transfer.
    """
    sent = 0
    for offset in range(0, size, packet_bytes):
        length = min(packet_bytes, size - offset)
        for _ in range(retry_limit + 1):
            sent += length
            if rng.random() >= loss:
                break
        else:
            return sent, False
    return sent, True


def _validate(config: dict[str, Any]) -> None:
    for key in INT_KEYS:
        check_int(config[key], key)
    for key in ("devices", "rounds", "packet_bytes"):
        check_int(config[key], key, 1)
    for key in ("packet_loss", "post_verify_fault_probability"):
        check_number(config[key], key, 0, 1)
    for key in ("bitrate_bps", "radio_power_mw"):
        check_number(config[key], key, 1e-6)


def simulate(old_path: str | Path, new_path: str | Path, config: dict[str, Any],
             output: str | Path) -> dict[str, Any]:
    """Simulate full-image and delta updates over a device fleet.

    Both modes use identically seeded random streams, so they see the same
    packet losses and injected faults. Writes ``config.json``,
    ``events.jsonl`` (one line per update event) and ``summary.json``.
    """
    old, new = Path(old_path).read_bytes(), Path(new_path).read_bytes()
    if not old or not new:
        raise TinyDeployError("firmware images must be nonempty")
    _validate(config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", config)
    result: dict[str, Any] = {"schema_version": 1, "evidence_kind": "simulation",
                              "old_sha256": digest(old), "new_sha256": digest(new),
                              "assumptions": ASSUMPTIONS, "modes": {}}
    log.info("ota start: %d devices x %d rounds, %d B packets, loss=%g, retries=%d, fault_p=%g, "
             "%g bps, %g mW, seed=%d", config["devices"], config["rounds"], config["packet_bytes"],
             config["packet_loss"], config["retry_limit"], config["post_verify_fault_probability"],
             config["bitrate_bps"], config["radio_power_mw"], config["seed"])
    log.info("images: old %d B sha256=%s, new %d B sha256=%s", len(old), digest(old)[:12],
             len(new), digest(new)[:12])
    progress_every = max(1, config["devices"] // 5)
    with (output / "events.jsonl").open("w") as events:
        for mode in ("full", "delta"):
            rng = random.Random(config["seed"])
            faults = random.Random(config["seed"] + 1)
            totals = {"events": 0, "committed": 0, "rolled_back": 0, "transfer_failed": 0,
                      "wire_bytes": 0, "transmitted_bytes": 0}
            for device_id in range(config["devices"]):
                device = Device(old, **{key: config[key] for key in LAYOUT_KEYS})
                for round_id in range(config["rounds"]):
                    desired = new if round_id % 2 == 0 else old
                    patch = make_delta(device.image, desired)
                    payload = wire(patch) if mode == "delta" else desired
                    sent, complete = transfer(len(payload), rng, config["packet_loss"],
                                              config["packet_bytes"], config["retry_limit"])
                    inject = faults.random() < config["post_verify_fault_probability"]
                    if complete:
                        expected = digest(desired)
                        status = device.update(patch if mode == "delta" else payload, expected,
                                               lambda data: digest(data) == expected,
                                               delta=mode == "delta", corrupt_after_verify=inject)
                    else:
                        status = "transfer_failed"
                    totals["events"] += 1
                    totals[status] = totals.get(status, 0) + 1
                    totals["wire_bytes"] += len(payload)
                    totals["transmitted_bytes"] += sent
                    if status != "committed":
                        log.info("[%s] device %4d round %d: %s (sent %d B%s)", mode, device_id, round_id,
                                 status, sent, ", fault injected" if complete and inject else "")
                    else:
                        log.debug("[%s] device %4d round %d: committed (sent %d B)", mode, device_id,
                                  round_id, sent)
                    events.write(json.dumps({
                        "mode": mode, "device": device_id, "round": round_id, "status": status,
                        "wire_bytes": len(payload), "sent_bytes": sent,
                        "post_verify_fault_injected": complete and inject,
                        "active_sha256": digest(device.image)}) + "\n")
                if (device_id + 1) % progress_every == 0:
                    log.info("[%s] progress %d/%d devices: committed=%d rolled_back=%d failed=%d on_air=%d B",
                             mode, device_id + 1, config["devices"], totals["committed"],
                             totals["rolled_back"], totals["transfer_failed"], totals["transmitted_bytes"])
            totals["radio_time_s"] = totals["transmitted_bytes"] * 8 / config["bitrate_bps"]
            totals["estimated_radio_energy_mj"] = totals["radio_time_s"] * config["radio_power_mw"]
            totals["rollback_rate"] = totals["rolled_back"] / totals["events"]
            totals["success_rate"] = totals["committed"] / totals["events"]
            result["modes"][mode] = totals
    full = result["modes"]["full"]["estimated_radio_energy_mj"]
    result["delta_energy_saving_fraction"] = 1 - result["modes"]["delta"]["estimated_radio_energy_mj"] / full
    write_json(output / "summary.json", result)
    log.info("ota done: full %.2f MB / delta %.2f MB on air, radio-energy saving %.1f%%",
             result["modes"]["full"]["transmitted_bytes"] / 1e6,
             result["modes"]["delta"]["transmitted_bytes"] / 1e6,
             100 * result["delta_energy_saving_fraction"])
    return result
