// evaluate.cpp
// Performance benchmark for wsq::BoundedWSQ and wsq::UnboundedWSQ.
//
// Usage:
//   ./evaluate [num_ops] [num_thieves] [r]
//
// Defaults: num_ops=1000000  num_thieves=3  r=20
//
// Each benchmark is run r+1 times. Round 0 is a warm-up excluded from stats.
// Results are shown for every combination of:
//   - Type:    int, int64_t
//   - LogSize: 8, 9, 10

#include "../wsq.hpp"

#include <atomic>
#include <barrier>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

using Clock    = std::chrono::steady_clock;
using Duration = std::chrono::duration<double>;

static double elapsed_sec(Clock::time_point s, Clock::time_point e) {
  return Duration(e - s).count();
}

static void print_result(const char* label, long ops,
                         const std::vector<double>& times) {
  int r = (int)times.size();
  std::vector<double> mops(r);
  for (int i = 0; i < r; ++i)
    mops[i] = static_cast<double>(ops) / times[i] / 1e6;

  double mean = std::accumulate(mops.begin(), mops.end(), 0.0) / r;
  double var  = 0.0;
  for (double v : mops) var += (v - mean) * (v - mean);
  double stddev = (r > 1) ? std::sqrt(var / (r - 1)) : 0.0;

  std::printf("  %-64s  %9.2f ± %7.2f Mops/s\n", label, mean, stddev);
}

template <typename BenchFn>
static void bench(const char* label, long ops, int r, BenchFn fn) {
  std::vector<double> times;
  times.reserve(r);
  for (int i = 0; i <= r; ++i) {
    double t = fn();
    if (i > 0) times.push_back(t);  // skip warm-up round
  }
  print_result(label, ops, times);
}

// ---------------------------------------------------------------------------
// BoundedWSQ benchmarks — templated on T and LogSize
// ---------------------------------------------------------------------------

template <typename T, size_t LogSize>
static double run_bounded_push_pop(long N) {
  wsq::BoundedWSQ<T, LogSize> q;
  std::vector<T> data(N, T{});

  const long cap = (long)q.capacity();
  auto start = Clock::now();
  for (long i = 0; i < N; ) {
    long batch = std::min(cap, N - i);
    for (long j = 0; j < batch; ++j) q.try_push(data[i + j]);
    for (long j = 0; j < batch; ++j) q.pop();
    i += batch;
  }
  return elapsed_sec(start, Clock::now());
}

template <typename T, size_t LogSize>
static double run_bounded_bulk(long N) {
  wsq::BoundedWSQ<T, LogSize> q;
  std::vector<T> data(N, T{});

  const long     cap   = (long)q.capacity();
  constexpr long BATCH = 64;
  auto start = Clock::now();
  for (long i = 0; i < N; ) {
    long window = std::min(cap, N - i);
    for (long j = 0; j + BATCH <= window; j += BATCH)
      q.try_bulk_push(data.data() + i + j, BATCH);
    while (q.pop()) {}
    i += window;
  }
  return elapsed_sec(start, Clock::now());
}

template <typename T, size_t LogSize>
static double run_bounded_concurrent(long N, int n_thieves) {
  wsq::BoundedWSQ<T, LogSize> q;
  std::vector<T> data(N, T{});

  const long         cap  = (long)q.capacity();
  std::atomic<long>  remaining{N};
  std::atomic<bool>  started{false};

  auto start = Clock::now();
  {
    std::vector<std::jthread> thieves;
    for (int i = 0; i < n_thieves; ++i) {
      thieves.emplace_back([&] {
        while (remaining.load(std::memory_order_acquire) > 0) {
          while (!started.load(std::memory_order_acquire) &&
                 remaining.load(std::memory_order_acquire) > 0)
            std::this_thread::yield();
          while (q.steal())
            remaining.fetch_sub(1, std::memory_order_relaxed);
        }
      });
    }

    for (long base = 0; base < N; base += cap) {
      long batch = std::min(cap, N - base);
      for (long j = 0; j < batch; ++j) q.try_push(data[base + j]);
      started.store(true, std::memory_order_release);
      while (q.pop()) remaining.fetch_sub(1, std::memory_order_relaxed);
      started.store(false, std::memory_order_release);
    }
  } // jthreads join here
  return elapsed_sec(start, Clock::now());
}

// ---------------------------------------------------------------------------
// UnboundedWSQ benchmarks — T is the element type; LogSize sets initial cap
// ---------------------------------------------------------------------------

template <typename T, size_t LogSize>
static double run_unbounded_push_pop(long N) {
  wsq::UnboundedWSQ<T> q(LogSize);
  std::vector<T> data(N, T{});

  auto start = Clock::now();
  for (long i = 0; i < N; ++i) q.push(data[i]);
  for (long i = 0; i < N; ++i) q.pop();
  return elapsed_sec(start, Clock::now());
}

template <typename T, size_t LogSize>
static double run_unbounded_bulk(long N) {
  wsq::UnboundedWSQ<T> q(LogSize);
  std::vector<T> data(N, T{});

  constexpr long BATCH = 64;
  auto start = Clock::now();
  for (long i = 0; i + BATCH <= N; i += BATCH)
    q.bulk_push(data.data() + i, BATCH);
  while (q.pop()) {}
  return elapsed_sec(start, Clock::now());
}

template <typename T, size_t LogSize>
static double run_unbounded_concurrent(long N, int n_thieves) {
  wsq::UnboundedWSQ<T> q(LogSize);
  std::vector<T> data(N, T{});
  for (long i = 0; i < N; ++i) q.push(data[i]);

  std::barrier sync(static_cast<std::ptrdiff_t>(n_thieves + 1));

  auto start = Clock::now();
  long owner = 0;
  {
    std::vector<std::jthread> thieves;
    for (int i = 0; i < n_thieves; ++i) {
      thieves.emplace_back([&] {
        sync.arrive_and_wait();
        while (true) {
          if      (q.steal())   {}
          else if (q.empty())   break;
          else                  std::this_thread::yield();
        }
      });
    }
    sync.arrive_and_wait();
    while (true) {
      if      (q.pop())   ++owner;
      else if (q.empty()) break;
      else                std::this_thread::yield();
    }
  } // jthreads join here
  return elapsed_sec(start, Clock::now());
}

// ---------------------------------------------------------------------------
// Run all 6 benchmarks for one (T, LogSize) combination
// ---------------------------------------------------------------------------

template <typename T, size_t LogSize>
static void bench_combination(const char* type_name, long N, int n_thieves, int r) {
  char label[128];

  std::printf("\n  -- BoundedWSQ<%s, %zu>  (capacity=%zu) --\n",
              type_name, LogSize, size_t{1} << LogSize);

  std::snprintf(label, sizeof(label),
    "BoundedWSQ<%s, %zu>   push+pop (owner only)", type_name, LogSize);
  bench(label, N * 2, r,
        [&]{ return run_bounded_push_pop<T, LogSize>(N); });

  std::snprintf(label, sizeof(label),
    "BoundedWSQ<%s, %zu>   bulk_push+pop (batch=64, owner only)", type_name, LogSize);
  bench(label, N * 2, r,
        [&]{ return run_bounded_bulk<T, LogSize>(N); });

  std::snprintf(label, sizeof(label),
    "BoundedWSQ<%s, %zu>   concurrent steal (%d thieves)", type_name, LogSize, n_thieves);
  bench(label, N, r,
        [&]{ return run_bounded_concurrent<T, LogSize>(N, n_thieves); });

  std::printf("\n  -- UnboundedWSQ<%s> (initial LogSize=%zu, capacity=%zu) --\n",
              type_name, LogSize, size_t{1} << LogSize);

  std::snprintf(label, sizeof(label),
    "UnboundedWSQ<%s, %zu> push+pop (owner only)", type_name, LogSize);
  bench(label, N * 2, r,
        [&]{ return run_unbounded_push_pop<T, LogSize>(N); });

  std::snprintf(label, sizeof(label),
    "UnboundedWSQ<%s, %zu> bulk_push+pop (batch=64, owner only)", type_name, LogSize);
  bench(label, N * 2, r,
        [&]{ return run_unbounded_bulk<T, LogSize>(N); });

  std::snprintf(label, sizeof(label),
    "UnboundedWSQ<%s, %zu> concurrent steal (%d thieves)", type_name, LogSize, n_thieves);
  bench(label, N, r,
        [&]{ return run_unbounded_concurrent<T, LogSize>(N, n_thieves); });
}

// Run all LogSizes (8-10) for a given type.
template <typename T>
static void bench_type(const char* type_name, long N, int n_thieves, int r) {
  std::printf("\n======== Type: %s ========\n", type_name);
  bench_combination<T,  8>(type_name, N, n_thieves, r);
  bench_combination<T,  9>(type_name, N, n_thieves, r);
  bench_combination<T, 10>(type_name, N, n_thieves, r);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
  long num_ops   = 1000000;
  int  n_thieves = 3;
  int  r         = 20;

  if (argc >= 2) num_ops   = std::atol(argv[1]);
  if (argc >= 3) n_thieves = std::atoi(argv[2]);
  if (argc >= 4) r         = std::atoi(argv[3]);

  if (num_ops <= 0 || n_thieves <= 0 || r <= 0) {
    std::fprintf(stderr,
      "Usage: %s [num_ops] [num_thieves] [r]\n"
      "  num_ops     > 0  (default 1000000)\n"
      "  num_thieves > 0  (default 3)\n"
      "  r           > 0  (default 10, timed rounds; +1 warm-up not counted)\n",
      argv[0]);
    return 1;
  }

  std::printf("Work-Stealing Queue Performance Evaluation\n");
  std::printf("  num_ops    = %ld\n",  num_ops);
  std::printf("  n_thieves  = %d\n",   n_thieves);
  std::printf("  r          = %d  (+ 1 warm-up round, not included in stats)\n", r);
  std::printf("  hw_threads = %u\n",   std::thread::hardware_concurrency());
  std::printf("  types      = int, int64_t\n");
  std::printf("  log_sizes  = 8 .. 10\n");
  std::printf("\n  %-64s  %s\n", "benchmark", "mean ± stddev Mops/s");
  std::printf("  %s\n", std::string(88, '-').c_str());

  bench_type<int>     ("int",     num_ops, n_thieves, r);
  bench_type<int64_t> ("int64_t", num_ops, n_thieves, r);

  std::printf("\nDone.\n");
  return 0;
}
