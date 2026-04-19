// unittest_wsq.cpp
// Comprehensive stress tests for wsq::BoundedWSQ and wsq::UnboundedWSQ.
//
// Test philosophy:
//   - Every concurrent test uses per-item atomic flags to detect both lost
//     items (flag never set) and duplicated items (flag set more than once).
//   - Tests are parameterised so they can be scaled up for longer soak runs.
//   - The uint64_t/init=1 correctness is verified both directly (index state
//     inspection via observable behaviour) and indirectly through tests that
//     hammer the empty-queue boundary across many cycles.

#include "../wsq.hpp"

#include <algorithm>
#include <atomic>
#include <barrier>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <numeric>
#include <random>
#include <string_view>
#include <thread>
#include <vector>

// ---------------------------------------------------------------------------
// Test harness
// ---------------------------------------------------------------------------

static int g_total  = 0;
static int g_passed = 0;
static int g_failed = 0;

#define CHECK(expr)                                                            \
  do {                                                                         \
    ++g_total;                                                                 \
    if (expr) {                                                                \
      ++g_passed;                                                              \
    } else {                                                                   \
      ++g_failed;                                                              \
      std::fprintf(stderr, "  FAIL  %s:%d  %s\n", __FILE__, __LINE__, #expr); \
    }                                                                          \
  } while (0)

static void run_suite(std::string_view name, std::function<void()> fn) {
  std::printf("[suite] %s\n", name.data());
  fflush(stdout);
  fn();
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Verify that every item in `seen` was observed exactly `expected_count` times.
static bool check_all_seen_exactly(const std::vector<std::atomic<int>>& seen,
                                   int expected_count, const char* ctx) {
  bool ok = true;
  for (size_t i = 0; i < seen.size(); ++i) {
    int v = seen[i].load(std::memory_order_relaxed);
    if (v != expected_count) {
      std::fprintf(stderr,
        "  FAIL  [%s] item[%zu] seen %d times (expected %d)\n",
        ctx, i, v, expected_count);
      ok = false;
    }
  }
  return ok;
}

// ============================================================================
// Section 1: uint64_t / init=1 correctness
// ============================================================================

// ---------------------------------------------------------------------------
// 1a. Pop on a brand-new queue must not corrupt _bottom.
//     We verify this by checking that the queue is fully usable after any
//     number of empty pop()/steal() calls on a fresh instance.
// ---------------------------------------------------------------------------
static void test_uint64_init_pop_safety() {
  run_suite("uint64/init=1: pop on empty does not corrupt _bottom", [] {
    // BoundedWSQ
    {
      wsq::BoundedWSQ<int*, 3> q;  // cap=8
      // Hammer empty pops — if _bottom wraps to UINT64_MAX this will break.
      for (int i = 0; i < 1000; ++i) {
        CHECK(q.pop()   == nullptr);
        CHECK(q.steal() == nullptr);
      }
      // Queue must still be fully functional
      int vals[8] = {1,2,3,4,5,6,7,8};
      for (auto& v : vals) CHECK(q.try_push(&v));
      CHECK(q.size() == 8);
      CHECK(!q.try_push(&vals[0]));  // must be full
      for (int i = 7; i >= 0; --i) CHECK(q.pop() == &vals[i]);
      CHECK(q.empty());
    }

    // UnboundedWSQ
    {
      wsq::UnboundedWSQ<int*> q;
      for (int i = 0; i < 1000; ++i) {
        CHECK(q.pop()   == nullptr);
        CHECK(q.steal() == nullptr);
      }
      int vals[8] = {1,2,3,4,5,6,7,8};
      for (auto& v : vals) q.push(&v);
      CHECK(q.size() == 8);
      for (int i = 7; i >= 0; --i) CHECK(q.pop() == &vals[i]);
      CHECK(q.empty());
    }
  });
}

// ---------------------------------------------------------------------------
// 1b. Alternate push/pop at the boundary: push 1, pop 1, repeat many times.
//     This keeps the queue oscillating between size 0 and size 1, exercising
//     the last-item CAS path and the empty-check path on every iteration.
//     Correctness: each pushed item must be the one popped.
// ---------------------------------------------------------------------------
static void test_uint64_oscillate_boundary() {
  run_suite("uint64/init=1: oscillate push/pop at size 0<->1", [] {
    constexpr int N = 500'000;

    // BoundedWSQ
    {
      wsq::BoundedWSQ<int*> q;
      std::vector<int> data(N);
      std::iota(data.begin(), data.end(), 0);
      bool ok = true;
      for (int i = 0; i < N; ++i) {
        CHECK(q.try_push(&data[i]));
        auto* got = q.pop();
        if (got != &data[i]) ok = false;
      }
      CHECK(ok);
      CHECK(q.empty());
    }

    // UnboundedWSQ
    {
      wsq::UnboundedWSQ<int*> q;
      std::vector<int> data(N);
      std::iota(data.begin(), data.end(), 0);
      bool ok = true;
      for (int i = 0; i < N; ++i) {
        q.push(&data[i]);
        auto* got = q.pop();
        if (got != &data[i]) ok = false;
      }
      CHECK(ok);
      CHECK(q.empty());
    }
  });
}

// ---------------------------------------------------------------------------
// 1c. Simulate the exact scenario that would fail with uint64 init=0:
//     call pop() on an empty queue, then verify _bottom recovered correctly
//     by doing a sequence of pushes that exactly matches capacity.
// ---------------------------------------------------------------------------
static void test_uint64_pop_then_fill() {
  run_suite("uint64/init=1: pop-on-empty then fill-to-capacity", [] {
    // BoundedWSQ cap=4
    {
      wsq::BoundedWSQ<int*, 2> q;
      CHECK(q.pop() == nullptr);   // _bottom goes 1->0->1 (safe with init=1)
      CHECK(q.pop() == nullptr);
      int v[4] = {10, 20, 30, 40};
      CHECK(q.try_push(&v[0]));
      CHECK(q.try_push(&v[1]));
      CHECK(q.try_push(&v[2]));
      CHECK(q.try_push(&v[3]));
      CHECK(q.size() == 4);
      CHECK(!q.try_push(&v[0]));   // full — if _bottom corrupted, might not return false
      CHECK(q.pop() == &v[3]);
      CHECK(q.pop() == &v[2]);
      CHECK(q.pop() == &v[1]);
      CHECK(q.pop() == &v[0]);
      CHECK(q.empty());
    }

    // UnboundedWSQ
    {
      wsq::UnboundedWSQ<int*> q;
      CHECK(q.pop() == nullptr);
      CHECK(q.pop() == nullptr);
      int v[4] = {10, 20, 30, 40};
      for (auto& x : v) q.push(&x);
      CHECK(q.size() == 4);
      CHECK(q.pop() == &v[3]);
      CHECK(q.pop() == &v[2]);
      CHECK(q.pop() == &v[1]);
      CHECK(q.pop() == &v[0]);
      CHECK(q.empty());
    }
  });
}

// ---------------------------------------------------------------------------
// 1d. Multi-cycle drain: fill, completely drain, repeat many times.
//     Detects any state that accumulates incorrectly across cycles.
//     Note: try_bulk_push may push fewer than BATCH when _cached_top is stale
//     after a steal cycle (conservative occupancy estimate). This is correct
//     behaviour -- we verify total items recovered per cycle, not exact push count.
// ---------------------------------------------------------------------------
static void test_uint64_multicycle_drain() {
  run_suite("uint64/init=1: multi-cycle fill+drain correctness", [] {
    constexpr int CYCLES = 200;
    constexpr int BATCH  = 64;

    // BoundedWSQ
    {
      wsq::BoundedWSQ<int*, 7> q;  // cap=128
      std::vector<int> data(BATCH);
      std::iota(data.begin(), data.end(), 0);
      std::vector<int*> ptrs(BATCH);
      for (int i = 0; i < BATCH; ++i) ptrs[i] = &data[i];

      for (int c = 0; c < CYCLES; ++c) {
        // Push as many as possible (may be < BATCH if _cached_top is stale)
        size_t pushed = q.try_bulk_push(ptrs.data(), BATCH);
        // If stale cache gave fewer slots, force-fill remaining one by one
        // (try_push refreshes on each full check)
        for (size_t i = pushed; i < BATCH; ++i) {
          CHECK(q.try_push(ptrs[i]));
        }
        CHECK(q.size() == BATCH);

        // Alternate pop (LIFO) and steal (FIFO) each cycle
        if (c % 2 == 0) {
          for (int i = BATCH - 1; i >= 0; --i) CHECK(q.pop() == &data[i]);
        } else {
          for (int i = 0; i < BATCH; ++i) CHECK(q.steal() == &data[i]);
        }
        CHECK(q.empty());
        CHECK(q.size() == 0);
      }
    }

    // UnboundedWSQ (bulk_push always pushes all N -- no partial push)
    {
      wsq::UnboundedWSQ<int*> q;
      std::vector<int> data(BATCH);
      std::iota(data.begin(), data.end(), 0);
      std::vector<int*> ptrs(BATCH);
      for (int i = 0; i < BATCH; ++i) ptrs[i] = &data[i];

      for (int c = 0; c < CYCLES; ++c) {
        q.bulk_push(ptrs.data(), BATCH);
        CHECK(q.size() == BATCH);
        if (c % 2 == 0) {
          for (int i = BATCH - 1; i >= 0; --i) CHECK(q.pop() == &data[i]);
        } else {
          for (int i = 0; i < BATCH; ++i) CHECK(q.steal() == &data[i]);
        }
        CHECK(q.empty());
      }
    }
  });
}

// ============================================================================
// Section 2: Single-threaded correctness
// ============================================================================

static void test_bounded_lifo_fifo_ordering() {
  run_suite("BoundedWSQ: strict LIFO pop / FIFO steal ordering", [] {
    wsq::BoundedWSQ<int*, 4> q;  // cap=16
    int vals[16];
    for (int i = 0; i < 16; ++i) vals[i] = i;

    for (int i = 0; i < 16; ++i) CHECK(q.try_push(&vals[i]));

    // pop() returns newest first (LIFO)
    for (int i = 15; i >= 8; --i) CHECK(q.pop() == &vals[i]);

    // steal() returns oldest first (FIFO) from the remaining 8
    for (int i = 0; i < 8; ++i) CHECK(q.steal() == &vals[i]);

    CHECK(q.empty());
  });
}

static void test_unbounded_lifo_fifo_ordering() {
  run_suite("UnboundedWSQ: strict LIFO pop / FIFO steal ordering", [] {
    wsq::UnboundedWSQ<int*> q;
    int vals[16];
    for (int i = 0; i < 16; ++i) vals[i] = i;

    for (int i = 0; i < 16; ++i) q.push(&vals[i]);

    for (int i = 15; i >= 8; --i) CHECK(q.pop() == &vals[i]);
    for (int i = 0;  i <  8; ++i) CHECK(q.steal() == &vals[i]);

    CHECK(q.empty());
  });
}

static void test_unbounded_resize_correctness() {
  run_suite("UnboundedWSQ: resize preserves all items in order", [] {
    // Start tiny, force many doublings
    wsq::UnboundedWSQ<int*> q(1);  // cap=2
    constexpr int N = 8192;
    std::vector<int> data(N);
    std::iota(data.begin(), data.end(), 0);

    for (auto& v : data) q.push(&v);
    CHECK(q.size()     == N);
    CHECK(q.capacity() >= (size_t)N);

    // LIFO pop — must come out in exact reverse order
    bool ok = true;
    for (int i = N - 1; i >= 0; --i) {
      if (q.pop() != &data[i]) { ok = false; break; }
    }
    CHECK(ok);
    CHECK(q.empty());
  });
}

static void test_bounded_bulk_push_correctness() {
  run_suite("BoundedWSQ: try_bulk_push partial/full semantics", [] {
    wsq::BoundedWSQ<int*, 3> q;  // cap=8
    int data[20];
    int* ptrs[20];
    for (int i = 0; i < 20; ++i) { data[i] = i; ptrs[i] = &data[i]; }

    // Push 5 — should all fit
    CHECK(q.try_bulk_push(ptrs, 5) == 5);
    CHECK(q.size() == 5);

    // Push 5 more — only 3 fit
    CHECK(q.try_bulk_push(ptrs + 5, 5) == 3);
    CHECK(q.size() == 8);

    // Full — returns 0
    CHECK(q.try_bulk_push(ptrs, 1) == 0);

    // Pop and verify LIFO order of all 8 pushed items
    int* expected[] = {
      ptrs[7], ptrs[6], ptrs[5],
      ptrs[4], ptrs[3], ptrs[2], ptrs[1], ptrs[0]
    };
    for (auto* e : expected) CHECK(q.pop() == e);
    CHECK(q.empty());
  });
}

static void test_unbounded_bulk_push_correctness() {
  run_suite("UnboundedWSQ: bulk_push with resize preserves order", [] {
    wsq::UnboundedWSQ<int*> q(1);  // cap=2, will resize
    constexpr int N = 1000;
    std::vector<int> data(N);
    std::iota(data.begin(), data.end(), 0);
    std::vector<int*> ptrs(N);
    for (int i = 0; i < N; ++i) ptrs[i] = &data[i];

    // Push in varying batch sizes to exercise array boundaries
    int pushed = 0;
    for (int batch : {7, 13, 50, 100, 200, 630}) {
      if (pushed + batch > N) batch = N - pushed;
      q.bulk_push(ptrs.data() + pushed, batch);
      pushed += batch;
      if (pushed >= N) break;
    }
    CHECK(q.size() == (size_t)pushed);

    bool ok = true;
    for (int i = pushed - 1; i >= 0; --i) {
      if (q.pop() != &data[i]) { ok = false; break; }
    }
    CHECK(ok);
    CHECK(q.empty());
  });
}

// ============================================================================
// Section 3: Concurrent stress — no item lost, no item duplicated
//
// Each item gets an atomic<int> counter. Any thread that claims an item
// increments counter[item_index]. At the end we verify every counter == 1.
// ============================================================================

// ---------------------------------------------------------------------------
// 3a. BoundedWSQ: owner pushes continuously, N thieves steal.
//     Owner also pops. Every item must be claimed exactly once.
// ---------------------------------------------------------------------------
static void stress_bounded_push_pop_steal(int n_items, int n_thieves) {
  char name[128];
  std::snprintf(name, sizeof(name),
    "BoundedWSQ stress: push+pop+steal  items=%d thieves=%d",
    n_items, n_thieves);

  run_suite(name, [&] {
    auto q = std::make_unique<wsq::BoundedWSQ<int*, 17>>(); // cap=128k
    std::vector<int> data(n_items);
    std::iota(data.begin(), data.end(), 0);
    std::vector<std::atomic<int>> seen(n_items);
    for (auto& s : seen) s.store(0, std::memory_order_relaxed);

    std::atomic<bool> done{false};
    std::barrier sync(static_cast<std::ptrdiff_t>(n_thieves + 1));

    {
      std::vector<std::jthread> thieves;
      for (int i = 0; i < n_thieves; ++i) {
        thieves.emplace_back([&] {
          sync.arrive_and_wait();
          while (true) {
            auto* item = q->steal();
            if (item != nullptr) {
              seen[item - data.data()].fetch_add(1, std::memory_order_relaxed);
            } else if (done.load(std::memory_order_acquire) && q->empty()) {
              break;
            } else {
              std::this_thread::yield();
            }
          }
        });
      }

      sync.arrive_and_wait();
      int idx = 0;
      while (idx < n_items) {
        int batch = std::min(64, n_items - idx);
        for (int i = 0; i < batch; ++i) {
          while (!q->try_push(&data[idx])) std::this_thread::yield();
          ++idx;
        }
        // Pop half the batch back (simulate real scheduler)
        for (int i = 0; i < batch / 2; ++i) {
          auto* item = q->pop();
          if (item) seen[item - data.data()].fetch_add(1, std::memory_order_relaxed);
        }
      }
      while (auto* item = q->pop()) {
        seen[item - data.data()].fetch_add(1, std::memory_order_relaxed);
      }
      done.store(true, std::memory_order_release);
    } // jthreads join here — all thieves done before CHECK

    CHECK(check_all_seen_exactly(seen, 1, name));
  });
}

// ---------------------------------------------------------------------------
// 3b. UnboundedWSQ: owner pushes+pops while thieves steal.
//     Pushes start from tiny capacity to stress the array swap under concurrency.
// ---------------------------------------------------------------------------
static void stress_unbounded_push_pop_steal(int n_items, int n_thieves) {
  char name[128];
  std::snprintf(name, sizeof(name),
    "UnboundedWSQ stress: push+pop+steal+resize  items=%d thieves=%d",
    n_items, n_thieves);

  run_suite(name, [&] {
    wsq::UnboundedWSQ<int*> q(2);  // start tiny (cap=4) to force resizes
    std::vector<int> data(n_items);
    std::iota(data.begin(), data.end(), 0);
    std::vector<std::atomic<int>> seen(n_items);
    for (auto& s : seen) s.store(0, std::memory_order_relaxed);

    std::atomic<bool> done{false};
    std::barrier sync(static_cast<std::ptrdiff_t>(n_thieves + 1));

    {
      std::vector<std::jthread> thieves;
      for (int i = 0; i < n_thieves; ++i) {
        thieves.emplace_back([&] {
          sync.arrive_and_wait();
          while (true) {
            auto* item = q.steal();
            if (item != nullptr) {
              seen[item - data.data()].fetch_add(1, std::memory_order_relaxed);
            } else if (done.load(std::memory_order_acquire) && q.empty()) {
              break;
            } else {
              std::this_thread::yield();
            }
          }
        });
      }

      sync.arrive_and_wait();
      // Build a pointer array so bulk_push receives int** -> int* elements
      std::vector<int*> ptrs(n_items);
      for (int i = 0; i < n_items; ++i) ptrs[i] = &data[i];

      int idx = 0;
      while (idx < n_items) {
        int batch = std::min(32, n_items - idx);
        q.bulk_push(ptrs.data() + idx, batch);
        idx += batch;
        for (int i = 0; i < batch / 2; ++i) {
          auto* item = q.pop();
          if (item) seen[item - data.data()].fetch_add(1, std::memory_order_relaxed);
        }
      }
      while (auto* item = q.pop()) {
        seen[item - data.data()].fetch_add(1, std::memory_order_relaxed);
      }
      done.store(true, std::memory_order_release);
    } // jthreads join here

    CHECK(check_all_seen_exactly(seen, 1, name));
  });
}

// ---------------------------------------------------------------------------
// 3c. Last-item race: hammer the single-element boundary.
//     One owner, one thief, each racing for the sole item every round.
//     Every item must be claimed exactly once.
// ---------------------------------------------------------------------------
static void stress_last_item_race(int rounds) {
  char name[128];
  std::snprintf(name, sizeof(name),
    "BoundedWSQ stress: last-item race  rounds=%d", rounds);

  run_suite(name, [&] {
    wsq::BoundedWSQ<int*> q;
    std::vector<int> data(rounds);
    std::iota(data.begin(), data.end(), 0);
    std::vector<std::atomic<int>> seen(rounds);
    for (auto& s : seen) s.store(0, std::memory_order_relaxed);

    std::atomic<bool> done{false};

    {
      std::jthread thief([&] {
        while (true) {
          auto* item = q.steal();
          if (item) {
            seen[item - data.data()].fetch_add(1, std::memory_order_relaxed);
          } else if (done.load(std::memory_order_acquire)) {
            break;
          }
        }
      });

      for (int r = 0; r < rounds; ++r) {
        while (!q.try_push(&data[r])) {}
        auto* item = q.pop();
        if (item) {
          seen[item - data.data()].fetch_add(1, std::memory_order_relaxed);
        }
      }
      done.store(true, std::memory_order_release);
    } // jthread joins here

    CHECK(check_all_seen_exactly(seen, 1, name));
  });
}

static void stress_unbounded_last_item_race(int rounds) {
  char name[128];
  std::snprintf(name, sizeof(name),
    "UnboundedWSQ stress: last-item race  rounds=%d", rounds);

  run_suite(name, [&] {
    wsq::UnboundedWSQ<int*> q;
    std::vector<int> data(rounds);
    std::iota(data.begin(), data.end(), 0);
    std::vector<std::atomic<int>> seen(rounds);
    for (auto& s : seen) s.store(0, std::memory_order_relaxed);

    std::atomic<bool> done{false};

    {
      std::jthread thief([&] {
        while (true) {
          auto* item = q.steal();
          if (item) {
            seen[item - data.data()].fetch_add(1, std::memory_order_relaxed);
          } else if (done.load(std::memory_order_acquire)) {
            break;
          }
        }
      });

      for (int r = 0; r < rounds; ++r) {
        q.push(&data[r]);
        auto* item = q.pop();
        if (item) {
          seen[item - data.data()].fetch_add(1, std::memory_order_relaxed);
        }
      }
      done.store(true, std::memory_order_release);
    } // jthread joins here

    CHECK(check_all_seen_exactly(seen, 1, name));
  });
}

// ---------------------------------------------------------------------------
// 3d. Multi-round stress: repeatedly fill and drain concurrently.
//     Each round the owner pushes N items; thieves steal; owner pops leftovers.
//     Items must each appear exactly once per round.
// ---------------------------------------------------------------------------
static void stress_multiround(int n_rounds, int n_per_round, int n_thieves) {
  char name[128];
  std::snprintf(name, sizeof(name),
    "BoundedWSQ stress: multi-round  rounds=%d per_round=%d thieves=%d",
    n_rounds, n_per_round, n_thieves);

  run_suite(name, [&] {
    auto q = std::make_unique<wsq::BoundedWSQ<int*, 17>>();
    std::vector<int> data(n_per_round);
    std::iota(data.begin(), data.end(), 0);
    std::vector<std::atomic<int>> seen(n_per_round);

    for (int r = 0; r < n_rounds; ++r) {
      for (auto& s : seen) s.store(0, std::memory_order_relaxed);

      std::atomic<bool> round_done{false};
      std::barrier sync(static_cast<std::ptrdiff_t>(n_thieves + 1));

      {
        std::vector<std::jthread> thieves;
        for (int i = 0; i < n_thieves; ++i) {
          thieves.emplace_back([&] {
            sync.arrive_and_wait();
            while (true) {
              auto* item = q->steal();
              if (item) {
                seen[item - data.data()].fetch_add(1, std::memory_order_relaxed);
              } else if (round_done.load(std::memory_order_acquire) && q->empty()) {
                break;
              } else {
                std::this_thread::yield();
              }
            }
          });
        }

        sync.arrive_and_wait();
        for (auto& v : data) {
          while (!q->try_push(&v)) std::this_thread::yield();
        }
        while (auto* item = q->pop()) {
          seen[item - data.data()].fetch_add(1, std::memory_order_relaxed);
        }
        round_done.store(true, std::memory_order_release);
      } // jthreads join here

      bool ok = check_all_seen_exactly(seen, 1, name);
      CHECK(ok);
      if (!ok) break;
    }
  });
}

// ---------------------------------------------------------------------------
// 3e. Randomised interleaving: owner randomly chooses push, pop, or bulk_push.
//     Thieves steal continuously. Every item must be claimed exactly once.
// ---------------------------------------------------------------------------
static void stress_random_interleaving(int n_items, int n_thieves, uint64_t seed) {
  char name[128];
  std::snprintf(name, sizeof(name),
    "UnboundedWSQ stress: random interleaving  items=%d thieves=%d seed=%llu",
    n_items, n_thieves, (unsigned long long)seed);

  run_suite(name, [&] {
    wsq::UnboundedWSQ<int*> q;
    std::vector<int> data(n_items);
    std::iota(data.begin(), data.end(), 0);
    std::vector<std::atomic<int>> seen(n_items);
    for (auto& s : seen) s.store(0, std::memory_order_relaxed);

    std::atomic<int>  remaining{n_items};
    std::atomic<bool> all_pushed{false};
    std::barrier sync(static_cast<std::ptrdiff_t>(n_thieves + 1));

    {
      std::vector<std::jthread> thieves;
      for (int i = 0; i < n_thieves; ++i) {
        thieves.emplace_back([&] {
          sync.arrive_and_wait();
          while (true) {
            auto* item = q.steal();
            if (item) {
              seen[item - data.data()].fetch_add(1, std::memory_order_relaxed);
              remaining.fetch_sub(1, std::memory_order_relaxed);
            } else if (all_pushed.load(std::memory_order_acquire) &&
                       remaining.load(std::memory_order_relaxed) == 0) {
              break;
            } else {
              std::this_thread::yield();
            }
          }
        });
      }

      sync.arrive_and_wait();
      std::mt19937_64 rng(seed);
      std::vector<int*> ptrs(n_items);
      for (int i = 0; i < n_items; ++i) ptrs[i] = &data[i];
      int pushed = 0;

      while (pushed < n_items || !q.empty()) {
        int choice = (int)(rng() % 3);
        if (choice == 0 && pushed < n_items) {
          q.push(&data[pushed++]);
        } else if (choice == 1 && pushed < n_items) {
          int batch = std::min((int)(rng() % 15 + 2), n_items - pushed);
          q.bulk_push(ptrs.data() + pushed, batch);
          pushed += batch;
        } else {
          auto* item = q.pop();
          if (item) {
            seen[item - data.data()].fetch_add(1, std::memory_order_relaxed);
            remaining.fetch_sub(1, std::memory_order_relaxed);
          }
        }
      }
      all_pushed.store(true, std::memory_order_release);
      while (auto* item = q.pop()) {
        seen[item - data.data()].fetch_add(1, std::memory_order_relaxed);
        remaining.fetch_sub(1, std::memory_order_relaxed);
      }
    } // jthreads join here

    CHECK(check_all_seen_exactly(seen, 1, name));
  });
}

// ---------------------------------------------------------------------------
// 3f. High-contention steal: many thieves, single-item pushes.
//     Maximises races on the _top CAS among thieves.
// ---------------------------------------------------------------------------
static void stress_high_contention_steal(int n_items, int n_thieves) {
  char name[128];
  std::snprintf(name, sizeof(name),
    "BoundedWSQ stress: high-contention steal  items=%d thieves=%d",
    n_items, n_thieves);

  run_suite(name, [&] {
    wsq::BoundedWSQ<int*> q;  // cap=256, deliberately small
    std::vector<int> data(n_items);
    std::iota(data.begin(), data.end(), 0);
    std::vector<std::atomic<int>> seen(n_items);
    for (auto& s : seen) s.store(0, std::memory_order_relaxed);

    std::atomic<bool> done{false};
    std::barrier sync(static_cast<std::ptrdiff_t>(n_thieves + 1));

    {
      std::vector<std::jthread> thieves;
      for (int i = 0; i < n_thieves; ++i) {
        thieves.emplace_back([&] {
          sync.arrive_and_wait();
          while (true) {
            auto* item = q.steal();
            if (item) {
              seen[item - data.data()].fetch_add(1, std::memory_order_relaxed);
            } else if (done.load(std::memory_order_acquire) && q.empty()) {
              break;
            }
            // no yield — maximise CAS contention among thieves
          }
        });
      }

      sync.arrive_and_wait();
      for (int i = 0; i < n_items; ++i) {
        while (!q.try_push(&data[i])) std::this_thread::yield();
      }
      while (auto* item = q.pop()) {
        seen[item - data.data()].fetch_add(1, std::memory_order_relaxed);
      }
      done.store(true, std::memory_order_release);
    } // jthreads join here

    CHECK(check_all_seen_exactly(seen, 1, name));
  });
}

// ---------------------------------------------------------------------------
// 3g. steal_with_feedback correctness under concurrency.
// ---------------------------------------------------------------------------
static void stress_steal_with_feedback(int n_items, int n_thieves) {
  char name[128];
  std::snprintf(name, sizeof(name),
    "BoundedWSQ stress: steal_with_feedback  items=%d thieves=%d",
    n_items, n_thieves);

  run_suite(name, [&] {
    auto q = std::make_unique<wsq::BoundedWSQ<int*, 17>>();
    std::vector<int> data(n_items);
    std::iota(data.begin(), data.end(), 0);
    std::vector<std::atomic<int>> seen(n_items);
    for (auto& s : seen) s.store(0, std::memory_order_relaxed);

    std::atomic<bool> done{false};
    std::barrier sync(static_cast<std::ptrdiff_t>(n_thieves + 1));

    {
      std::vector<std::jthread> thieves;
      for (int i = 0; i < n_thieves; ++i) {
        thieves.emplace_back([&] {
          sync.arrive_and_wait();
          size_t empty_steals = 0;
          while (true) {
            auto* item = q->steal_with_feedback(empty_steals);
            if (item) {
              seen[item - data.data()].fetch_add(1, std::memory_order_relaxed);
            } else if (done.load(std::memory_order_acquire) && q->empty()) {
              break;
            } else {
              std::this_thread::yield();
            }
          }
        });
      }

      sync.arrive_and_wait();
      for (auto& v : data) {
        while (!q->try_push(&v)) std::this_thread::yield();
      }
      while (auto* item = q->pop()) {
        seen[item - data.data()].fetch_add(1, std::memory_order_relaxed);
      }
      done.store(true, std::memory_order_release);
    } // jthreads join here

    CHECK(check_all_seen_exactly(seen, 1, name));
  });
}

// ============================================================================
// main
// ============================================================================

int main(int argc, char* argv[]) {
  // Usage: ./unittest_wsq [scale] [n_thieves]
  //
  //   scale      - multiplier for item counts and round counts (default: 1)
  //                scale=1  ~ a few seconds per configuration
  //                scale=10 ~ soak / overnight run
  //
  //   n_thieves  - number of concurrent thief threads for Section 3 (default: auto)
  //                0 or omitted → auto: clamp(hw_threads-1, 1, 16)
  //                Any positive value overrides auto-detection.
  //
  // CTest registers a grid of (scale, n_thieves) combinations.
  // Sections 1 and 2 are deterministic and only run when n_thieves == 0
  // (i.e. the first entry in the grid) to avoid redundant repetition.

  int scale     = 1;
  int n_thieves = 0;  // 0 = auto

  if (argc >= 2) scale     = std::atoi(argv[1]);
  if (argc >= 3) n_thieves = std::atoi(argv[2]);

  if (scale     < 1) scale     = 1;
  if (n_thieves < 0) n_thieves = 0;

  const int HW = (int)std::thread::hardware_concurrency();

  // Resolve auto
  if (n_thieves == 0) {
    n_thieves = std::max(1, std::min(HW - 1, 16));
  }

  std::printf("Work-Stealing Queue Unit Tests\n");
  std::printf("  scale=%d  n_thieves=%d  hw_threads=%d\n\n",
              scale, n_thieves, HW);

  // ------------------------------------------------------------------
  // Section 1 & 2: deterministic, single-threaded — run unconditionally.
  // (CTest will only schedule these once via the n_thieves=0 variant,
  //  but running them every time is cheap and catches regressions.)
  // ------------------------------------------------------------------
  std::printf("-- Section 1: uint64_t / init=1 correctness --\n");
  test_uint64_init_pop_safety();
  test_uint64_oscillate_boundary();
  test_uint64_pop_then_fill();
  test_uint64_multicycle_drain();

  std::printf("\n-- Section 2: Single-threaded correctness --\n");
  test_bounded_lifo_fifo_ordering();
  test_unbounded_lifo_fifo_ordering();
  test_unbounded_resize_correctness();
  test_bounded_bulk_push_correctness();
  test_unbounded_bulk_push_correctness();

  // ------------------------------------------------------------------
  // Section 3: concurrent stress — parameterised by scale & n_thieves
  // ------------------------------------------------------------------
  std::printf("\n-- Section 3: Concurrent stress (scale=%d, thieves=%d) --\n",
              scale, n_thieves);

  const int N      = 200'000 * scale;
  const int ROUNDS = 20      * scale;
  const int LAST   = 100'000 * scale;

  stress_bounded_push_pop_steal   (N,                  n_thieves);
  stress_unbounded_push_pop_steal (N,                  n_thieves);
  stress_last_item_race           (LAST);
  stress_unbounded_last_item_race (LAST);
  stress_multiround               (ROUNDS, N / ROUNDS, n_thieves);
  stress_random_interleaving      (N,      n_thieves,  0xdeadbeefcafe1234ULL);
  stress_random_interleaving      (N,      n_thieves,  0x0123456789abcdefULL);
  stress_high_contention_steal    (50'000 * scale,     n_thieves);
  stress_steal_with_feedback      (N,                  n_thieves);

  std::printf("\n%d / %d tests passed", g_passed, g_total);
  if (g_failed) {
    std::printf("  (%d FAILED)\n", g_failed);
    return EXIT_FAILURE;
  }
  std::printf("  -- all OK\n");
  return EXIT_SUCCESS;
}
