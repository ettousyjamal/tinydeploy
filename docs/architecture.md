# Architecture

TinyDeploy is a host-side orchestrator. It does not implement compression
algorithms, inference kernels or a bootloader; it connects the tools that do,
checks their outputs against hardware budgets and records every decision.

## Pipeline

| Stage | Module | Input | Output |
|---|---|---|---|
| 1. Candidate generation | `grid`, `compression` | Trained model; bit-widths and pruning ratios; target support | One model file per supported configuration; skipped configurations with reasons |
| 2. Profiling | `profiles` | Model; profiler report or board timing CSV; validation score | Normalized profile JSON bound to the model hash |
| 3. Filtering and selection | `selection` | Profiles; target budgets; minimum score | `results.json` with every candidate, its rejection reasons, the feasible and Pareto sets, the selection |
| 4. Target build hand-off | `build` | Selected model; board build command | `model_data.h`, `firmware.bin`, build log with firmware hash |
| 5. OTA preparation | `delta`, `ota` | Previous and new images; fleet configuration | Hash-bound delta; per-event log; summary |

## Responsibility boundaries (paper, Table II)

| Module | Provided by external tools | Provided by TinyDeploy |
|---|---|---|
| Candidate generation | Edge Impulse / EON Tuner in the reported workflow; TensorFlow Lite converter in the open adapter | Grid expansion (Algorithm 1), support checks, logged invocation of the compressor |
| Evaluation and selection | On-board timing (Edge Impulse runner or `firmware/tinydeploy_benchmark.h`); RAM and flash from the Edge Impulse build report | Profile schema and validation, hash binding, budget filter, energy estimate, Pareto set, selection |
| Target build | TFLite Micro, CMSIS-NN and the board toolchain (graph optimizations, kernels, arena planning) | C array emission, shell-free build invocation, build log, firmware size and hash |
| OTA preparation and recovery | Radio stack and bootloader on the device | Delta codec, A/B update logic, fleet simulation and its logs |

## Selection rule (Algorithm 1)

For each bit-width `b` in {8, 4} and pruning ratio `p` in {0, 0.3, 0.5, 0.7}:

1. skip `(b, p)` if the target cannot run `b`-bit kernels (`supported_bit_widths`);
2. compress, then profile the candidate (external commands, each logged);
3. compute energy per inference `E = latency x current x voltage`;
4. reject it if score < minimum, peak SRAM > SRAM budget, model + firmware
   reserve > flash, or latency > maximum latency (all violated budgets are listed).

Among feasible candidates, the highest score is selected; ties go to lower
energy, then to smaller model size. The Pareto set (score up; size, SRAM,
latency down) is reported alongside, since the finite grid does not guarantee
global optimality.

## OTA update logic

Each device has two equal firmware slots after the bootloader, metadata and
scratch regions. An update is downloaded into the inactive slot as a full image
or a delta, its SHA-256 is verified, a post-boot self-test runs, and the device
commits or rolls back to the previous image. The simulator repeats this for a
fleet with seeded packet loss, bounded retries and injected post-verification
faults, using identical random streams for the full-image and delta modes.
Radio energy is `transmitted bits / bitrate x radio power`; headers, ACKs,
duty-cycle limits and flash/CPU energy are excluded and are listed in each
summary's `assumptions`.
