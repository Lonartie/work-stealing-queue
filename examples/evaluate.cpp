// evaluate.cpp
// Comprehensive performance benchmark for wsq::BoundedWSQ and wsq::UnboundedWSQ.
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
//
// Benchmark scenarios
// -------------------
//  1. push+pop          Owner fills then drains one capacity-window at a time.
//                       Measures amortised push+pop throughput with no contention.
//
//  2. interleaved       Owner alternates push(1) / pop(1). Stresses round-trip
//                       latency of the index-update cycle rather than throughput.
//
//  3. burst             Owner repeatedly fills to a fraction of capacity then
//                       drains. Parameterised at 25 %, 50 %, 75 % fill.
//                       Shows how performance varies with occupancy level.
//
//  4. bulk_push+pop     Like push+pop but using try_bulk_push / bulk_push with
//                       batch=64. Tests amortisation of the fence+store.
//
//  5. bulk_var          bulk_push with varying batch sizes (1, 16, 64, 256).
//                       Shows how amortisation scales with batch size.
//
//  6. concurrent        Owner pre-fills then owner+thieves race to drain.
//                       Classic "filled bag" scenario.
//
//  7. continuous        Owner pushes non-stop while thieves steal concurrently.
//                       The true steady-state work-stealing hot path.
//
//  8. sparse_steal      Owner pushes slowly (one item at a time with a yield
//                       between) while thieves poll via steal_with_feedback.
//                       Tests the back-off feedback path under a sparse queue.
//
//  9. bulk_concurrent   Owner bulk-pushes one large batch, then owner+thieves
//                       race to drain. Isolates bulk-push amortisation from
//                       the drain contention measured in scenario 6.
//
// 10. multi_queue       N+1 threads each own one queue; each thread pushes to
//                       its own queue and randomly steals from a neighbour.
//                       The topology used by most real thread-pool schedulers.
//
// 11. resize            UnboundedWSQ only. Starts at LogSize=4 (16 slots) and
//                       pushes N items, forcing repeated doublings. Measures
//                       amortised cost of resize + _garbage accumulation.

#include "../wsq.hpp"

#include <atomic>
#include <barrier>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>

using Clock    = std::chrono::steady_clock;
using Duration = std::chrono::duration<double>;

static double elapsed_sec(Clock::time_point s, Clock::time_point e) {
  return Duration(e - s).count();
}

// ---------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------

// Label column width — wide enough for the longest int64_t label.
static constexpr int LABEL_W = 70;

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

  std::printf("  %-*s  %9.2f ± %7.2f Mops/s\n", LABEL_W, label, mean, stddev);
  fflush(stdout);
}

template <typename BenchFn>
static void bench(const char* label, long ops, int r, BenchFn fn) {
  std::vector<double> times;
  times.reserve(r);
  for (int i = 0; i <= r; ++i) {
    double t = fn();
    if (i > 0) times.push_back(t);
  }
  print_result(label, ops, times);
}

// ---------------------------------------------------------------------------
// Scenario 1: push+pop (owner only, capacity-windowed)
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
static double run_unbounded_push_pop(long N) {
  wsq::UnboundedWSQ<T> q(LogSize);
  std::vector<T> data(N, T{});

  auto start = Clock::now();
  for (long i = 0; i < N; ++i) q.push(data[i]);
  for (long i = 0; i < N; ++i) q.pop();
  return elapsed_sec(start, Clock::now());
}

// ---------------------------------------------------------------------------
// Scenario 2: interleaved push+pop (owner only, one at a time)
// ---------------------------------------------------------------------------

template <typename T, size_t LogSize>
static double run_bounded_interleaved(long N) {
  wsq::BoundedWSQ<T, LogSize> q;
  std::vector<T> data(N, T{});

  auto start = Clock::now();
  for (long i = 0; i < N; ++i) {
    q.try_push(data[i]);
    q.pop();
  }
  return elapsed_sec(start, Clock::now());
}

template <typename T, size_t LogSize>
static double run_unbounded_interleaved(long N) {
  wsq::UnboundedWSQ<T> q(LogSize);
  std::vector<T> data(N, T{});

  auto start = Clock::now();
  for (long i = 0; i < N; ++i) {
    q.push(data[i]);
    q.pop();
  }
  return elapsed_sec(start, Clock::now());
}

// ---------------------------------------------------------------------------
// Scenario 3: burst (owner only, fill to fill_pct% of capacity, drain, repeat)
// ---------------------------------------------------------------------------

template <typename T, size_t LogSize>
static double run_bounded_burst(long N, int fill_pct) {
  wsq::BoundedWSQ<T, LogSize> q;
  std::vector<T> data(N, T{});
  const long cap   = (long)q.capacity();
  const long burst = std::max(1L, cap * fill_pct / 100);

  auto start = Clock::now();
  for (long i = 0; i < N; ) {
    long batch = std::min(burst, N - i);
    for (long j = 0; j < batch; ++j) q.try_push(data[i + j]);
    for (long j = 0; j < batch; ++j) q.pop();
    i += batch;
  }
  return elapsed_sec(start, Clock::now());
}

template <typename T, size_t LogSize>
static double run_unbounded_burst(long N, int fill_pct) {
  wsq::UnboundedWSQ<T> q(LogSize);
  std::vector<T> data(N, T{});
  const long cap   = (long)(size_t{1} << LogSize);
  const long burst = std::max(1L, cap * fill_pct / 100);

  auto start = Clock::now();
  for (long i = 0; i < N; ) {
    long batch = std::min(burst, N - i);
    for (long j = 0; j < batch; ++j) q.push(data[i + j]);
    for (long j = 0; j < batch; ++j) q.pop();
    i += batch;
  }
  return elapsed_sec(start, Clock::now());
}

// ---------------------------------------------------------------------------
// Scenario 4: bulk_push+pop (owner only, batch=64)
// ---------------------------------------------------------------------------

template <typename T, size_t LogSize>
static double run_bounded_bulk(long N) {
  wsq::BoundedWSQ<T, LogSize> q;
  std::vector<T> data(N, T{});
  const long cap        = (long)q.capacity();
  constexpr long BATCH  = 64;

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

// ---------------------------------------------------------------------------
// Scenario 5: bulk_push with varying batch sizes (owner only)
// ---------------------------------------------------------------------------

template <typename T, size_t LogSize>
static double run_bounded_bulk_var(long N, long batch_size) {
  wsq::BoundedWSQ<T, LogSize> q;
  std::vector<T> data(N, T{});
  const long cap = (long)q.capacity();

  auto start = Clock::now();
  for (long i = 0; i < N; ) {
    long window = std::min(cap, N - i);
    for (long j = 0; j + batch_size <= window; j += batch_size)
      q.try_bulk_push(data.data() + i + j, (size_t)batch_size);
    while (q.pop()) {}
    i += window;
  }
  return elapsed_sec(start, Clock::now());
}

template <typename T, size_t LogSize>
static double run_unbounded_bulk_var(long N, long batch_size) {
  wsq::UnboundedWSQ<T> q(LogSize);
  std::vector<T> data(N, T{});

  auto start = Clock::now();
  for (long i = 0; i + batch_size <= N; i += batch_size)
    q.bulk_push(data.data() + i, (size_t)batch_size);
  while (q.pop()) {}
  return elapsed_sec(start, Clock::now());
}

// ---------------------------------------------------------------------------
// Scenario 6: concurrent (pre-fill then owner+thieves drain)
// ---------------------------------------------------------------------------

template <typename T, size_t LogSize>
static double run_bounded_concurrent(long N, int n_thieves) {
  wsq::BoundedWSQ<T, LogSize> q;
  std::vector<T> data(N, T{});
  const long cap = (long)q.capacity();
  std::atomic<long> remaining{N};
  std::atomic<bool> started{false};

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

template <typename T, size_t LogSize>
static double run_unbounded_concurrent(long N, int n_thieves) {
  wsq::UnboundedWSQ<T> q(LogSize);
  std::vector<T> data(N, T{});
  for (long i = 0; i < N; ++i) q.push(data[i]);

  std::barrier sync(static_cast<std::ptrdiff_t>(n_thieves + 1));
  auto start = Clock::now();
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
      if      (q.pop())   {}
      else if (q.empty()) break;
      else                std::this_thread::yield();
    }
  } // jthreads join here
  return elapsed_sec(start, Clock::now());
}

// ---------------------------------------------------------------------------
// Scenario 7: continuous (owner pushes non-stop, thieves steal concurrently)
// ---------------------------------------------------------------------------

template <typename T, size_t LogSize>
static double run_bounded_continuous(long N, int n_thieves) {
  wsq::BoundedWSQ<T, LogSize> q;
  std::vector<T> data(N, T{});

  // remaining tracks items not yet claimed by anyone
  std::atomic<long> remaining{N};
  std::atomic<bool> done{false};

  auto start = Clock::now();
  {
    std::vector<std::jthread> thieves;
    for (int i = 0; i < n_thieves; ++i) {
      thieves.emplace_back([&] {
        while (!done.load(std::memory_order_acquire) ||
               remaining.load(std::memory_order_relaxed) > 0) {
          if (q.steal())
            remaining.fetch_sub(1, std::memory_order_relaxed);
          else
            std::this_thread::yield();
        }
      });
    }

    // Owner pushes continuously; pop when the queue is full
    for (long i = 0; i < N; ) {
      if (q.try_push(data[i])) {
        ++i;
      } else {
        // queue full — pop one to make room (owner-side work)
        if (q.pop()) remaining.fetch_sub(1, std::memory_order_relaxed);
        else         std::this_thread::yield();
      }
    }
    // Drain whatever is left
    while (remaining.load(std::memory_order_relaxed) > 0) {
      if (q.pop()) remaining.fetch_sub(1, std::memory_order_relaxed);
      else         std::this_thread::yield();
    }
    done.store(true, std::memory_order_release);
  } // jthreads join here
  return elapsed_sec(start, Clock::now());
}

template <typename T, size_t LogSize>
static double run_unbounded_continuous(long N, int n_thieves) {
  wsq::UnboundedWSQ<T> q(LogSize);
  std::vector<T> data(N, T{});

  std::atomic<long> remaining{N};
  std::atomic<bool> done{false};

  auto start = Clock::now();
  {
    std::vector<std::jthread> thieves;
    for (int i = 0; i < n_thieves; ++i) {
      thieves.emplace_back([&] {
        while (!done.load(std::memory_order_acquire) ||
               remaining.load(std::memory_order_relaxed) > 0) {
          if (q.steal())
            remaining.fetch_sub(1, std::memory_order_relaxed);
          else
            std::this_thread::yield();
        }
      });
    }

    // Owner pushes all N items, popping occasionally to let thieves
    // stay busy rather than spinning on a queue that's growing faster
    // than they can steal.
    for (long i = 0; i < N; ++i) {
      q.push(data[i]);
      // Every 64 pushes let owner pop some to balance with thieves
      if ((i & 63) == 63) {
        if (q.pop()) remaining.fetch_sub(1, std::memory_order_relaxed);
      }
    }
    while (remaining.load(std::memory_order_relaxed) > 0) {
      if (q.pop()) remaining.fetch_sub(1, std::memory_order_relaxed);
      else         std::this_thread::yield();
    }
    done.store(true, std::memory_order_release);
  } // jthreads join here
  return elapsed_sec(start, Clock::now());
}

// ---------------------------------------------------------------------------
// Scenario 8: sparse_steal (owner pushes slowly, thieves use steal_with_feedback)
// ---------------------------------------------------------------------------

template <typename T, size_t LogSize>
static double run_bounded_sparse(long N, int n_thieves) {
  long sparse_N = std::min(N, 10000L);
  wsq::BoundedWSQ<T, LogSize> q;
  std::vector<T> data(sparse_N, T{});

  std::atomic<long> remaining{sparse_N};
  std::atomic<bool> done{false};

  auto start = Clock::now();
  {
    std::vector<std::jthread> thieves;
    for (int i = 0; i < n_thieves; ++i) {
      thieves.emplace_back([&] {
        size_t empty_steals = 0;
        while (!done.load(std::memory_order_acquire) ||
               remaining.load(std::memory_order_relaxed) > 0) {
          if (q.steal_with_feedback(empty_steals))
            remaining.fetch_sub(1, std::memory_order_relaxed);
          else if (empty_steals > 16)
            std::this_thread::yield(); // back off after repeated misses
        }
      });
    }

    // Owner pushes one item at a time, yielding between each push to keep
    // the queue sparse so thieves frequently see an empty queue.
    // Cap at 10k items: the yield() makes this inherently slow.
    long sparse_N = std::min(N, 10000L);
    for (long i = 0; i < sparse_N; ++i) {
      while (!q.try_push(data[i])) std::this_thread::yield();
      std::this_thread::yield(); // deliberate slow production
    }
    done.store(true, std::memory_order_release);
  } // jthreads join here
  return elapsed_sec(start, Clock::now());
}

template <typename T, size_t LogSize>
static double run_unbounded_sparse(long N, int n_thieves) {
  long sparse_N = std::min(N, 10000L);
  wsq::UnboundedWSQ<T> q(LogSize);
  std::vector<T> data(sparse_N, T{});

  std::atomic<long> remaining{sparse_N};
  std::atomic<bool> done{false};

  auto start = Clock::now();
  {
    std::vector<std::jthread> thieves;
    for (int i = 0; i < n_thieves; ++i) {
      thieves.emplace_back([&] {
        size_t empty_steals = 0;
        while (!done.load(std::memory_order_acquire) ||
               remaining.load(std::memory_order_relaxed) > 0) {
          if (q.steal_with_feedback(empty_steals))
            remaining.fetch_sub(1, std::memory_order_relaxed);
          else if (empty_steals > 16)
            std::this_thread::yield();
        }
      });
    }

    long sparse_N = std::min(N, 10000L);
    for (long i = 0; i < sparse_N; ++i) {
      q.push(data[i]);
      std::this_thread::yield();
    }
    done.store(true, std::memory_order_release);
  } // jthreads join here
  return elapsed_sec(start, Clock::now());
}

// ---------------------------------------------------------------------------
// Scenario 9: bulk_concurrent (bulk-fill once, then owner+thieves drain)
// ---------------------------------------------------------------------------

template <typename T, size_t LogSize>
static double run_bounded_bulk_concurrent(long N, int n_thieves) {
  // For each capacity window: owner bulk-pushes in 64-item chunks, then a
  // per-window barrier releases owner+thieves to drain together via empty().
  // Using empty() as the drain signal avoids any counter arithmetic and is
  // exactly as safe here as in the unbounded version, because the queue is
  // only written (pushed) before the barrier and only read (pop/steal) after.
  wsq::BoundedWSQ<T, LogSize> q;
  std::vector<T> data(N, T{});
  const long cap = (long)q.capacity();

  // reusable barrier: owner + n_thieves arrive twice per window (fill, drain)
  std::barrier bar(static_cast<std::ptrdiff_t>(n_thieves + 1));

  auto start = Clock::now();
  {
    std::vector<std::jthread> thieves;
    for (int i = 0; i < n_thieves; ++i) {
      thieves.emplace_back([&] {
        for (long base = 0; base < N; base += cap) {
          bar.arrive_and_wait(); // wait for owner to finish bulk-push
          while (true) {
            if      (q.steal())   {}
            else if (q.empty())   break;
            else                  std::this_thread::yield();
          }
          bar.arrive_and_wait(); // signal drain complete for this window
        }
      });
    }

    for (long base = 0; base < N; base += cap) {
      long window = std::min(cap, N - base);
      for (long j = 0; j + 64 <= window; j += 64)
        q.try_bulk_push(data.data() + base + j, 64);
      bar.arrive_and_wait(); // signal bulk-push done, release thieves
      // Owner drains its own side while thieves steal
      while (true) {
        if      (q.pop())   {}
        else if (q.empty()) break;
        else                std::this_thread::yield();
      }
      bar.arrive_and_wait(); // wait for all thieves to finish this window
    }
  } // jthreads join here
  return elapsed_sec(start, Clock::now());
}

template <typename T, size_t LogSize>
static double run_unbounded_bulk_concurrent(long N, int n_thieves) {
  wsq::UnboundedWSQ<T> q(LogSize);
  std::vector<T> data(N, T{});

  // Single bulk push of all N items, then concurrent drain
  q.bulk_push(data.data(), (size_t)N);

  std::barrier sync(static_cast<std::ptrdiff_t>(n_thieves + 1));
  auto start = Clock::now();
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
      if      (q.pop())   {}
      else if (q.empty()) break;
      else                std::this_thread::yield();
    }
  } // jthreads join here
  return elapsed_sec(start, Clock::now());
}

// ---------------------------------------------------------------------------
// Scenario 10: multi_queue (N+1 threads, each owns one queue, steals from neighbour)
// ---------------------------------------------------------------------------

template <typename T, size_t LogSize>
static double run_bounded_multi_queue(long N, int n_threads) {
  // Each thread owns one queue. It pushes its share of N items into its own
  // queue (in capacity-sized windows, since BoundedWSQ is fixed-size) while
  // concurrently popping from its own queue and stealing from its neighbour.
  // This is the actual topology used by real work-stealing schedulers.
  int nq = n_threads;
  std::vector<wsq::BoundedWSQ<T, LogSize>> queues(nq);
  std::vector<T> data(N, T{});

  // Per-thread item counts (last thread gets the remainder)
  long per_q = N / nq;
  std::atomic<long> remaining{N};
  std::barrier sync(static_cast<std::ptrdiff_t>(nq));

  auto start = Clock::now();
  {
    std::vector<std::jthread> threads;
    for (int qi = 0; qi < nq; ++qi) {
      threads.emplace_back([&, qi] {
        sync.arrive_and_wait();
        int    victim  = (qi + 1) % nq;
        long   my_start = (long)qi * per_q;
        long   my_end   = (qi == nq - 1) ? N : my_start + per_q;
        long   pushed   = my_start; // index into data[]

        while (remaining.load(std::memory_order_relaxed) > 0) {
          // Push one item into own queue if we still have items to push
          // and the queue has room
          if (pushed < my_end) {
            if (queues[qi].try_push(data[pushed]))
              ++pushed;
          }
          // Pop from own queue (LIFO — what we just pushed)
          if (queues[qi].pop()) {
            remaining.fetch_sub(1, std::memory_order_relaxed);
          }
          // Steal from neighbour when own queue is empty
          else if (queues[victim].steal()) {
            remaining.fetch_sub(1, std::memory_order_relaxed);
          }
          else if (pushed >= my_end) {
            std::this_thread::yield();
          }
        }
      });
    }
  } // jthreads join here
  return elapsed_sec(start, Clock::now());
}

template <typename T, size_t LogSize>
static double run_unbounded_multi_queue(long N, int n_threads) {
  // UnboundedWSQ is non-copyable and non-movable (contains atomics), so we
  // heap-allocate each queue and hold it via unique_ptr.
  int nq = n_threads;
  std::vector<std::unique_ptr<wsq::UnboundedWSQ<T>>> queues;
  queues.reserve(nq);
  for (int i = 0; i < nq; ++i)
    queues.push_back(std::make_unique<wsq::UnboundedWSQ<T>>(LogSize));

  std::vector<T> data(N, T{});
  long per_q = N / nq;

  std::atomic<long> remaining{N};
  std::barrier sync(static_cast<std::ptrdiff_t>(nq));

  auto start = Clock::now();
  {
    std::vector<std::jthread> threads;
    for (int qi = 0; qi < nq; ++qi) {
      threads.emplace_back([&, qi] {
        sync.arrive_and_wait();
        int  victim   = (qi + 1) % nq;
        long my_start = (long)qi * per_q;
        long my_end   = (qi == nq - 1) ? N : my_start + per_q;
        long pushed   = my_start;

        while (remaining.load(std::memory_order_relaxed) > 0) {
          // Push one item into own queue if we still have items to push
          if (pushed < my_end)
            queues[qi]->push(data[pushed++]);
          // Pop from own queue
          if (queues[qi]->pop()) {
            remaining.fetch_sub(1, std::memory_order_relaxed);
          }
          // Steal from neighbour when own queue is empty
          else if (queues[victim]->steal()) {
            remaining.fetch_sub(1, std::memory_order_relaxed);
          }
          else if (pushed >= my_end) {
            std::this_thread::yield();
          }
        }
      });
    }
  } // jthreads join here
  return elapsed_sec(start, Clock::now());
}

// ---------------------------------------------------------------------------
// Scenario 11: resize (UnboundedWSQ only — starts tiny, forces many doublings)
// ---------------------------------------------------------------------------

template <typename T>
static double run_unbounded_resize(long N) {
  // Start at capacity 16 (LogSize=4) to guarantee many doublings
  wsq::UnboundedWSQ<T> q(4);
  std::vector<T> data(N, T{});

  auto start = Clock::now();
  for (long i = 0; i < N; ++i) q.push(data[i]);
  for (long i = 0; i < N; ++i) q.pop();
  return elapsed_sec(start, Clock::now());
}

// ---------------------------------------------------------------------------
// bench_combination: run all scenarios for one (T, LogSize)
// ---------------------------------------------------------------------------

template <typename T, size_t LogSize>
static void bench_combination(const char* type_name, long N, int n_thieves, int r) {
  char label[160];
  const int nt = n_thieves;
  const int nq = n_thieves + 1; // total threads for multi_queue

  // ---- BoundedWSQ ----
  std::printf("\n  -- BoundedWSQ<%s, %zu>  (capacity=%zu) --\n",
              type_name, LogSize, size_t{1} << LogSize);
  fflush(stdout);

  // 1. push+pop
  std::snprintf(label, sizeof(label),
    "BoundedWSQ<%s,%2zu> [1] push+pop (owner only)", type_name, LogSize);
  bench(label, N * 2, r, [&]{ return run_bounded_push_pop<T, LogSize>(N); });

  // 2. interleaved
  std::snprintf(label, sizeof(label),
    "BoundedWSQ<%s,%2zu> [2] interleaved push/pop (owner only)", type_name, LogSize);
  bench(label, N * 2, r, [&]{ return run_bounded_interleaved<T, LogSize>(N); });

  // 3. burst 25 / 50 / 75
  for (int pct : {25, 50, 75}) {
    std::snprintf(label, sizeof(label),
      "BoundedWSQ<%s,%2zu> [3] burst fill=%d%% (owner only)", type_name, LogSize, pct);
    bench(label, N * 2, r, [&, pct]{ return run_bounded_burst<T, LogSize>(N, pct); });
  }

  // 4. bulk_push+pop (batch=64)
  std::snprintf(label, sizeof(label),
    "BoundedWSQ<%s,%2zu> [4] bulk_push+pop batch=64 (owner only)", type_name, LogSize);
  bench(label, N * 2, r, [&]{ return run_bounded_bulk<T, LogSize>(N); });

  // 5. bulk variable batch sizes
  for (long bs : {1L, 16L, 64L, 256L}) {
    std::snprintf(label, sizeof(label),
      "BoundedWSQ<%s,%2zu> [5] bulk batch=%-3ld (owner only)", type_name, LogSize, bs);
    bench(label, N * 2, r, [&, bs]{ return run_bounded_bulk_var<T, LogSize>(N, bs); });
  }

  // 6. concurrent (pre-fill + drain)
  std::snprintf(label, sizeof(label),
    "BoundedWSQ<%s,%2zu> [6] concurrent drain (%d thieves)", type_name, LogSize, nt);
  bench(label, N, r, [&]{ return run_bounded_concurrent<T, LogSize>(N, nt); });

  // 7. continuous push+steal
  std::snprintf(label, sizeof(label),
    "BoundedWSQ<%s,%2zu> [7] continuous push+steal (%d thieves)", type_name, LogSize, nt);
  bench(label, N, r, [&]{ return run_bounded_continuous<T, LogSize>(N, nt); });

  // 8. sparse steal with feedback
  std::snprintf(label, sizeof(label),
    "BoundedWSQ<%s,%2zu> [8] sparse steal_with_feedback (%d thieves)", type_name, LogSize, nt);
  bench(label, N, r, [&]{ return run_bounded_sparse<T, LogSize>(N, nt); });

  // 9. bulk_concurrent
  std::snprintf(label, sizeof(label),
    "BoundedWSQ<%s,%2zu> [9] bulk+concurrent drain (%d thieves)", type_name, LogSize, nt);
  bench(label, N, r, [&]{ return run_bounded_bulk_concurrent<T, LogSize>(N, nt); });

  // 10. multi_queue
  std::snprintf(label, sizeof(label),
    "BoundedWSQ<%s,%2zu>[10] multi_queue (%d threads)", type_name, LogSize, nq);
  bench(label, N, r, [&]{ return run_bounded_multi_queue<T, LogSize>(N, nq); });

  // ---- UnboundedWSQ ----
  std::printf("\n  -- UnboundedWSQ<%s> (initial LogSize=%zu, capacity=%zu) --\n",
              type_name, LogSize, size_t{1} << LogSize);
  fflush(stdout);

  // 1. push+pop
  std::snprintf(label, sizeof(label),
    "UnboundedWSQ<%s,%2zu> [1] push+pop (owner only)", type_name, LogSize);
  bench(label, N * 2, r, [&]{ return run_unbounded_push_pop<T, LogSize>(N); });

  // 2. interleaved
  std::snprintf(label, sizeof(label),
    "UnboundedWSQ<%s,%2zu> [2] interleaved push/pop (owner only)", type_name, LogSize);
  bench(label, N * 2, r, [&]{ return run_unbounded_interleaved<T, LogSize>(N); });

  // 3. burst 25 / 50 / 75
  for (int pct : {25, 50, 75}) {
    std::snprintf(label, sizeof(label),
      "UnboundedWSQ<%s,%2zu> [3] burst fill=%d%% (owner only)", type_name, LogSize, pct);
    bench(label, N * 2, r, [&, pct]{ return run_unbounded_burst<T, LogSize>(N, pct); });
  }

  // 4. bulk_push+pop (batch=64)
  std::snprintf(label, sizeof(label),
    "UnboundedWSQ<%s,%2zu> [4] bulk_push+pop batch=64 (owner only)", type_name, LogSize);
  bench(label, N * 2, r, [&]{ return run_unbounded_bulk<T, LogSize>(N); });

  // 5. bulk variable batch sizes
  for (long bs : {1L, 16L, 64L, 256L}) {
    std::snprintf(label, sizeof(label),
      "UnboundedWSQ<%s,%2zu> [5] bulk batch=%-3ld (owner only)", type_name, LogSize, bs);
    bench(label, N * 2, r, [&, bs]{ return run_unbounded_bulk_var<T, LogSize>(N, bs); });
  }

  // 6. concurrent (pre-fill + drain)
  std::snprintf(label, sizeof(label),
    "UnboundedWSQ<%s,%2zu> [6] concurrent drain (%d thieves)", type_name, LogSize, nt);
  bench(label, N, r, [&]{ return run_unbounded_concurrent<T, LogSize>(N, nt); });

  // 7. continuous push+steal
  std::snprintf(label, sizeof(label),
    "UnboundedWSQ<%s,%2zu> [7] continuous push+steal (%d thieves)", type_name, LogSize, nt);
  bench(label, N, r, [&]{ return run_unbounded_continuous<T, LogSize>(N, nt); });

  // 8. sparse steal with feedback
  std::snprintf(label, sizeof(label),
    "UnboundedWSQ<%s,%2zu> [8] sparse steal_with_feedback (%d thieves)", type_name, LogSize, nt);
  bench(label, N, r, [&]{ return run_unbounded_sparse<T, LogSize>(N, nt); });

  // 9. bulk_concurrent
  std::snprintf(label, sizeof(label),
    "UnboundedWSQ<%s,%2zu> [9] bulk+concurrent drain (%d thieves)", type_name, LogSize, nt);
  bench(label, N, r, [&]{ return run_unbounded_bulk_concurrent<T, LogSize>(N, nt); });

  // 10. multi_queue
  std::snprintf(label, sizeof(label),
    "UnboundedWSQ<%s,%2zu>[10] multi_queue (%d threads)", type_name, LogSize, nq);
  bench(label, N, r, [&]{ return run_unbounded_multi_queue<T, LogSize>(N, nq); });

  // 11. resize (UnboundedWSQ only — LogSize ignored, always starts at 4)
  std::snprintf(label, sizeof(label),
    "UnboundedWSQ<%s,  *>[11] resize stress (start LogSize=4)", type_name);
  bench(label, N * 2, r, [&]{ return run_unbounded_resize<T>(N); });
}

template <typename T>
static void bench_type(const char* type_name, long N, int n_thieves, int r) {
  std::printf("\n======== Type: %s ========\n", type_name);
  fflush(stdout);
  bench_combination<T,  8>(type_name, N, n_thieves, r);
  bench_combination<T,  9>(type_name, N, n_thieves, r);
  bench_combination<T, 10>(type_name, N, n_thieves, r);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
  long num_ops   = 1'000'000;
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
      "  r           > 0  (default 20, timed rounds; +1 warm-up not counted)\n",
      argv[0]);
    return 1;
  }

  const int nq = n_thieves + 1;
  std::printf("Work-Stealing Queue — Comprehensive Performance Evaluation\n");
  std::printf("  num_ops    = %ld\n",  num_ops);
  std::printf("  n_thieves  = %d\n",   n_thieves);
  std::printf("  n_threads  = %d  (for multi_queue scenario)\n", nq);
  std::printf("  r          = %d  (+ 1 warm-up, not in stats)\n", r);
  std::printf("  hw_threads = %u\n",   std::thread::hardware_concurrency());
  std::printf("  types      = int, int64_t\n");
  std::printf("  log_sizes  = 8, 9, 10\n");
  std::printf("\n  Scenarios:\n");
  std::printf("   [1] push+pop          owner-only, capacity-windowed\n");
  std::printf("   [2] interleaved       owner-only, push(1)+pop(1) alternating\n");
  std::printf("   [3] burst             owner-only, fill to 25/50/75%% then drain\n");
  std::printf("   [4] bulk_push+pop     owner-only, bulk batch=64\n");
  std::printf("   [5] bulk_var          owner-only, bulk batch=1/16/64/256\n");
  std::printf("   [6] concurrent        pre-fill then owner+thieves drain\n");
  std::printf("   [7] continuous        owner pushes non-stop, thieves steal\n");
  std::printf("   [8] sparse_steal      slow owner, thieves use steal_with_feedback\n");
  std::printf("   [9] bulk_concurrent   bulk-fill then owner+thieves drain\n");
  std::printf("  [10] multi_queue       each thread owns+steals across %d queues\n", nq);
  std::printf("  [11] resize            UnboundedWSQ only, start LogSize=4, force doublings\n");
  std::printf("\n  %-*s  %26s\n", LABEL_W, "benchmark", "mean ± stddev Mops/s");
  std::printf("  %s\n", std::string(LABEL_W + 30, '-').c_str());

  bench_type<int>     ("int",     num_ops, n_thieves, r);
  bench_type<int64_t> ("int64_t", num_ops, n_thieves, r);

  std::printf("\nDone.\n");
  return 0;
}
