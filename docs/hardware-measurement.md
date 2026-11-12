# On-board latency measurement

TinyDeploy accepts two kinds of hardware profiles: Edge Impulse profiling
reports (`profiler_estimate`) and on-board measurements (`measured`). In the
paper, latency was measured on the boards with the Edge Impulse impulse
runner. This page describes the repeated-timing protocol supported by
`firmware/tinydeploy_benchmark.h`, which adds warm-up, repetitions and raw
samples so that run-to-run variation can be reported.

## Protocol

1. Build the board firmware with TFLite Micro and CMSIS-NN and the selected
   `model_data.h`; record compiler, version and flags.
2. Fix the core clock (e.g. 80 MHz on STM32L476, 64 MHz on nRF52840) and
   enable a cycle-accurate timer (DWT `CYCCNT` on Cortex-M4).
3. Call `tinydeploy::Benchmark` with at least 1 warm-up and 30 timed
   repetitions. It first runs the golden self-test (a fixed input whose output
   must match a stored reference), then the warm-up, then the timed loop,
   emitting `sample_index,elapsed_ticks,timer_hz` per inference over UART.
4. Save the CSV and a protocol file:

```json
{"warmup": 5, "repetitions": 30, "timer": "DWT CYCCNT",
 "compiler": "arm-none-eabi-gcc 13.2", "flags": "-O3 -mcpu=cortex-m4 -mfpu=fpv4-sp-d16",
 "clock_hz": 80000000}
```

5. Import it as a measured profile:

```bash
tinydeploy import-profile benchmark timing.csv --protocol protocol.json \
    --model model.tflite --target stm32l476 --metric accuracy --score 0.912 \
    --peak-sram 61440 --output profile.json
```

The profile stores every raw sample, their mean, minimum, maximum and (after
validation) standard deviation, so reviewers can recompute the statistics.

## Edge Impulse reports

Edge Impulse device profiling returns, per device, `timePerInferenceMs` and
RAM/ROM for the EON compiler and TFLM. Save the JSON and convert it with:

```bash
tinydeploy import-profile edge-impulse report.json --model model.tflite \
    --target stm32l476 --metric accuracy --score 0.912 --engine eon --output profile.json
```
