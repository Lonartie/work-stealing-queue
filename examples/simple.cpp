// simple.cpp
// Demonstrates basic usage of wsq::BoundedWSQ and wsq::UnboundedWSQ.

#include "../wsq.hpp"

#include <thread>
#include <vector>
#include <atomic>
#include <cstdio>

// ---------------------------------------------------------------------------
// BoundedWSQ example
// ---------------------------------------------------------------------------
static void example_bounded() {
  std::printf("=== BoundedWSQ (capacity=256, LogSize=8) ===\n");

  wsq::BoundedWSQ<int*> q;  // 256-slot bounded queue
  std::printf("  capacity : %zu\n", q.capacity());
  std::printf("  empty    : %s\n", q.empty() ? "true" : "false");

  // Owner pushes tasks
  int tasks[5] = {10, 20, 30, 40, 50};
  for (auto& t : tasks) q.try_push(&t);
  std::printf("  pushed 5 tasks, size=%zu\n", q.size());

  // Owner pops (LIFO — gets most recently pushed first)
  std::printf("  owner pops (LIFO): ");
  while (auto* item = q.pop()) std::printf("%d ", *item);
  std::printf("\n");

  // Refill and let a thief steal (FIFO — gets oldest first)
  for (auto& t : tasks) q.try_push(&t);
  std::printf("  thief steals (FIFO): ");
  while (auto* item = q.steal()) std::printf("%d ", *item);
  std::printf("\n\n");
}

// ---------------------------------------------------------------------------
// UnboundedWSQ example
// ---------------------------------------------------------------------------
static void example_unbounded() {
  std::printf("=== UnboundedWSQ (initial capacity=16, LogSize=4) ===\n");

  wsq::UnboundedWSQ<int*> q(4);
  std::printf("  initial capacity : %zu\n", q.capacity());

  // Push more than initial capacity to trigger resize
  std::vector<int> tasks(20);
  for (int i = 0; i < 20; ++i) {
    tasks[i] = i * 100;
    q.push(&tasks[i]);
  }
  std::printf("  pushed 20 tasks (triggers resize)\n");
  std::printf("  capacity after growth : %zu\n", q.capacity());
  std::printf("  size                  : %zu\n\n", q.size());

  // Drain with pop
  std::printf("  popping all (LIFO): ");
  while (auto* item = q.pop()) std::printf("%d ", *item);
  std::printf("\n\n");
}

// ---------------------------------------------------------------------------
// Concurrent example: 1 owner + 3 thieves
// ---------------------------------------------------------------------------
static void example_concurrent() {
  std::printf("=== Concurrent: 1 owner pushes + 3 thieves steal ===\n");

  constexpr int N = 10'000;
  wsq::UnboundedWSQ<int*> q;

  std::vector<int> data(N);
  for (int i = 0; i < N; ++i) data[i] = i;

  std::atomic<int> stolen{0};

  auto thief_fn = [&] {
    int local = 0;
    while (true) {
      if (q.steal() != nullptr) {
        ++local;
      } else if (q.empty()) {
        break;
      }
    }
    stolen.fetch_add(local, std::memory_order_relaxed);
  };

  // Pre-push all tasks so thieves have work immediately
  for (auto& v : data) q.push(&v);

  int owner_popped = 0;
  {
    std::vector<std::jthread> thieves;
    for (int i = 0; i < 3; ++i) thieves.emplace_back(thief_fn);

    while (auto* item = q.pop()) {
      (void)item;
      ++owner_popped;
      if (q.empty()) break;
    }
    // Drain any remainder
    while (q.pop() != nullptr) ++owner_popped;
  } // jthreads join here — stolen count is stable before we print

  std::printf("  total tasks   : %d\n", N);
  std::printf("  owner popped  : %d\n", owner_popped);
  std::printf("  thieves stole : %d\n", stolen.load());
  std::printf("  accounted for : %d  (%s)\n\n",
    owner_popped + stolen.load(),
    (owner_popped + stolen.load() == N) ? "OK" : "MISMATCH");
}

// ---------------------------------------------------------------------------
// bulk_push example
// ---------------------------------------------------------------------------
static void example_bulk_push() {
  std::printf("=== bulk_push / try_bulk_push ===\n");

  // UnboundedWSQ bulk_push
  wsq::UnboundedWSQ<int*> uq(2);  // tiny start
  std::vector<int> data(100);
  std::vector<int*> ptrs(100);
  for (int i = 0; i < 100; ++i) { data[i] = i; ptrs[i] = &data[i]; }

  uq.bulk_push(ptrs.data(), 100);
  std::printf("  UnboundedWSQ bulk_push(100): size=%zu capacity=%zu\n",
    uq.size(), uq.capacity());

  // BoundedWSQ try_bulk_push (may be partial)
  wsq::BoundedWSQ<int*, 3> bq;  // capacity=8
  size_t pushed = bq.try_bulk_push(ptrs.data(), 100);
  std::printf("  BoundedWSQ  try_bulk_push(100) into cap=8: pushed=%zu\n\n",
    pushed);
}

int main() {
  example_bounded();
  example_unbounded();
  example_concurrent();
  example_bulk_push();
  return 0;
}
