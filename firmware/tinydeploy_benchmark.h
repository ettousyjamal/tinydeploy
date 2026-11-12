// TinyDeploy on-board latency benchmark helper (header-only, C++11).
//
// Measurement protocol: run the golden self-test once, perform `warmup`
// untimed inferences, then time `repetitions` inferences with a hardware
// counter (e.g. the Cortex-M DWT cycle counter) and emit one CSV row per
// sample:  sample_index,elapsed_ticks,timer_hz
//
// The CSV is converted into a measured TinyDeploy profile with
//   tinydeploy import-profile benchmark timing.csv --protocol protocol.json ...
// where protocol.json records warm-up, repetitions, timer, compiler, flags
// and clock frequency.
//
// The board project supplies the timer, the inference call (e.g. a TFLM
// MicroInterpreter::Invoke wrapper), the golden test and the CSV output.
#ifndef TINYDEPLOY_BENCHMARK_H
#define TINYDEPLOY_BENCHMARK_H

#include <stddef.h>
#include <stdint.h>

namespace tinydeploy {

typedef uint32_t (*ReadTicks)();
typedef bool (*Invoke)();
typedef bool (*GoldenTest)();
typedef void (*EmitCsv)(size_t sample, uint32_t elapsed_ticks, uint32_t timer_hz);

struct BenchmarkConfig {
  size_t warmup;       // untimed inferences before measurement (>= 1)
  size_t repetitions;  // timed inferences (>= 30)
  uint32_t timer_hz;   // tick frequency of ReadTicks
};

// Returns false if the configuration is invalid, the golden test fails, an
// inference fails, or the timer is too coarse to resolve one inference.
// One timed inference must be shorter than one counter wrap-around.
inline bool Benchmark(const BenchmarkConfig& config, ReadTicks ticks, Invoke invoke,
                      GoldenTest golden, EmitCsv emit) {
  if (!ticks || !invoke || !golden || !emit || !config.timer_hz ||
      config.warmup < 1 || config.repetitions < 30) {
    return false;
  }
  if (!golden()) return false;
  for (size_t i = 0; i < config.warmup; ++i) {
    if (!invoke()) return false;
  }
  for (size_t i = 0; i < config.repetitions; ++i) {
    const uint32_t begin = ticks();
    const bool ok = invoke();
    const uint32_t end = ticks();
    if (!ok) return false;
    const uint32_t elapsed = end - begin;  // unsigned arithmetic handles one wrap
    if (!elapsed) return false;
    emit(i, elapsed, config.timer_hz);     // output stays outside the timed region
  }
  return true;
}

}  // namespace tinydeploy

#endif  // TINYDEPLOY_BENCHMARK_H
