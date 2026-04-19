# work-stealing-queue

A single-header, lock-free, C++20 work-stealing deque with two variants:

| Class | Capacity | Push behaviour |
|---|---|---|
| `wsq::BoundedWSQ<T, LogSize>` | Fixed at compile time (2^LogSize) | `try_push` returns `false` when full |
| `wsq::UnboundedWSQ<T>` | Grows dynamically via array doubling | `push` always succeeds |

Both classes implement the Chase-Lev work-stealing deque described in:

> *Correct and Efficient Work-Stealing for Weak Memory Models*  
> Lê, Pop, Cohen, Nardelli — PPoPP 2013  
> https://www.di.ens.fr/~zappa/readings/ppopp13.pdf

## Requirements

- C++20 compiler (GCC ≥ 10, Clang ≥ 11, MSVC ≥ 19.29)
- CMake ≥ 3.18

## Quick start

```cpp
#include "wsq.hpp"

// Bounded queue (256 slots by default)
wsq::BoundedWSQ<int*> bq;
int x = 42;
bq.try_push(&x);
int* item = bq.pop();   // owner: LIFO
item = bq.steal();      // any thread: FIFO

// Unbounded queue (grows automatically)
wsq::UnboundedWSQ<int*> uq;
uq.push(&x);
item = uq.pop();
item = uq.steal();
```

## API

### BoundedWSQ\<T, LogSize\>

```cpp
bool        try_push(T item)                       // owner only
size_t      try_bulk_push(I first, size_t N)       // owner only
value_type  pop()                                  // owner only (LIFO)
value_type  steal()                                // any thread (FIFO)
value_type  steal_with_feedback(size_t& n_empty)   // any thread (FIFO)
bool        empty()    const noexcept
size_t      size()     const noexcept
size_t      capacity() const noexcept              // compile-time constant
```

### UnboundedWSQ\<T\>

```cpp
void        push(T item)                           // owner only
void        bulk_push(I first, size_t N)           // owner only
value_type  pop()                                  // owner only (LIFO)
value_type  steal()                                // any thread (FIFO)
value_type  steal_with_feedback(size_t& n_empty)   // any thread (FIFO)
bool        empty()    const noexcept
size_t      size()     const noexcept
size_t      capacity() const noexcept
```

`value_type` is `T` (with `nullptr` for empty) for pointer types, and
`std::optional<T>` (with `std::nullopt` for empty) for value types.

## Build

```bash
cd work-stealing-queue/
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=20
make -j$(nproc)
```

## Run unit tests

```bash
cd build
make test                      # run full grid (scale x thieves combinations)
ctest --output-on-failure      # same, with output on failure
ctest -R "scale=1"             # quick pass only
ctest -R "scale=4"             # thorough pass only
ctest -R "thieves=1"           # single-threaded stress only
```

The test grid is controlled by two CMake cache variables:

```bash
cmake .. -DWSQ_TEST_SCALES="1;4" -DWSQ_TEST_THIEVES="1;2;4;8;16"
```

## Run examples

```bash
./examples/simple                          # usage walkthrough
./examples/evaluate                        # default: 1M ops, 3 thieves, r=10
./examples/evaluate 5000000 7              # 5M ops, 7 thieves, r=10
./examples/evaluate 5000000 7 20           # 5M ops, 7 thieves, r=20 rounds
```

Each benchmark runs `r+1` rounds; round 0 is a warm-up and is excluded from
the reported mean ± stddev.

### Sample evaluate output

```
Work-Stealing Queue Performance Evaluation
  num_ops    = 1000000
  n_thieves  = 3
  r          = 10  (+ 1 warm-up round, not included in stats)
  hw_threads = 16

  benchmark                                                mean ± stddev Mops/s
  --------------------------------------------------------------------------------
  BoundedWSQ   push+pop (owner only)                      342.51 ±   4.12 Mops/s
  BoundedWSQ   bulk_push+pop (batch=64, owner only)       491.07 ±   6.83 Mops/s
  BoundedWSQ   concurrent steal (3 thieves)                97.34 ±   2.01 Mops/s

  UnboundedWSQ push+pop (owner only)                      253.18 ±   3.55 Mops/s
  UnboundedWSQ bulk_push+pop (batch=64, owner only)       379.44 ±   5.22 Mops/s
  UnboundedWSQ concurrent steal (3 thieves)                88.91 ±   1.74 Mops/s
```

## Compile-time knobs

| Macro | Default | Meaning |
|---|---|---|
| `WSQ_CACHELINE_SIZE` | `64` | Cache line size in bytes |
| `WSQ_DEFAULT_BOUNDED_LOG_SIZE` | `8` | Default log2 capacity for BoundedWSQ (256 slots) |
| `WSQ_DEFAULT_UNBOUNDED_LOG_SIZE` | `10` | Default initial log2 capacity for UnboundedWSQ (1024 slots) |

## License

MIT — see [LICENSE](LICENSE).
