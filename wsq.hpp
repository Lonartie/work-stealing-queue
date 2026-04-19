#pragma once

/**
@file wsq-original.hpp
@brief Original work-stealing queue — standalone, no optimizations.

Faithful port of the Taskflow wsq.hpp baseline to a self-contained header:
  - Removed all Taskflow dependencies (macros.hpp, traits.hpp)
  - Replaced TF_ macros with WSQ_ macros
  - Replaced namespace tf with namespace wsq
  - int64_t indices initialized to 0 (original behavior)
  - No cached-index optimization
  - _array shares a cache line with _bottom (original layout)

Algorithm: "Correct and Efficient Work-Stealing for Weak Memory Models"
  Lê, Pop, Cohen, Nardelli — PPoPP 2013
  https://www.di.ens.fr/~zappa/readings/ppopp13.pdf
*/

#include <atomic>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <type_traits>
#include <vector>

#ifndef WSQ_CACHELINE_SIZE
  #define WSQ_CACHELINE_SIZE 64
#endif

#ifndef WSQ_DEFAULT_BOUNDED_LOG_SIZE
  #define WSQ_DEFAULT_BOUNDED_LOG_SIZE 8
#endif

#ifndef WSQ_DEFAULT_UNBOUNDED_LOG_SIZE
  #define WSQ_DEFAULT_UNBOUNDED_LOG_SIZE 10
#endif

namespace wsq {

// ----------------------------------------------------------------------------
// UnboundedWSQ
// ----------------------------------------------------------------------------

template <typename T>
class UnboundedWSQ {

  struct Array {

    size_t C;
    size_t M;
    std::atomic<T>* S;

    explicit Array(size_t c) :
      C {c},
      M {c-1},
      S {new std::atomic<T>[C]} {
    }

    ~Array() {
      delete [] S;
    }

    size_t capacity() const noexcept {
      return C;
    }

    void push(int64_t i, T o) noexcept {
      S[i & M].store(o, std::memory_order_relaxed);
    }

    T pop(int64_t i) noexcept {
      return S[i & M].load(std::memory_order_relaxed);
    }

    Array* resize(int64_t b, int64_t t) {
      Array* ptr = new Array{2*C};
      for(int64_t i=t; i!=b; ++i) {
        ptr->push(i, pop(i));
      }
      return ptr;
    }

    Array* resize(int64_t b, int64_t t, size_t N) {
      Array* ptr = new Array{std::bit_ceil(C + N)};
      for(int64_t i=t; i!=b; ++i) {
        ptr->push(i, pop(i));
      }
      return ptr;
    }
  };

  // Original layout: _array shares _bottom's cache line (no alignment)
  alignas(WSQ_CACHELINE_SIZE) std::atomic<int64_t> _top;
  alignas(WSQ_CACHELINE_SIZE) std::atomic<int64_t> _bottom;
  std::atomic<Array*> _array;
  std::vector<Array*> _garbage;

  Array* _resize_array(Array* a, int64_t b, int64_t t);
  Array* _resize_array(Array* a, int64_t b, int64_t t, size_t N);

public:

  using value_type = std::conditional_t<std::is_pointer_v<T>, T, std::optional<T>>;

  explicit UnboundedWSQ(int64_t LogSize = WSQ_DEFAULT_UNBOUNDED_LOG_SIZE);
  ~UnboundedWSQ();

  UnboundedWSQ(const UnboundedWSQ&)            = delete;
  UnboundedWSQ& operator=(const UnboundedWSQ&) = delete;

  bool   empty()    const noexcept;
  size_t size()     const noexcept;
  size_t capacity() const noexcept;

  void push(T item);

  template <typename I>
  void bulk_push(I first, size_t N);

  value_type pop();
  value_type steal();
  value_type steal_with_feedback(size_t& num_empty_steals);

  static constexpr value_type empty_value() noexcept {
    if constexpr (std::is_pointer_v<T>) return T{nullptr};
    else                                return std::optional<T>{std::nullopt};
  }
};

// --- UnboundedWSQ implementation ---

template <typename T>
UnboundedWSQ<T>::UnboundedWSQ(int64_t LogSize) {
  _top.store(0, std::memory_order_relaxed);
  _bottom.store(0, std::memory_order_relaxed);
  _array.store(new Array{(size_t{1} << LogSize)}, std::memory_order_relaxed);
  _garbage.reserve(32);
}

template <typename T>
UnboundedWSQ<T>::~UnboundedWSQ() {
  for(auto a : _garbage) delete a;
  delete _array.load();
}

template <typename T>
bool UnboundedWSQ<T>::empty() const noexcept {
  int64_t t = _top.load(std::memory_order_relaxed);
  int64_t b = _bottom.load(std::memory_order_relaxed);
  return (b <= t);
}

template <typename T>
size_t UnboundedWSQ<T>::size() const noexcept {
  int64_t t = _top.load(std::memory_order_relaxed);
  int64_t b = _bottom.load(std::memory_order_relaxed);
  return static_cast<size_t>(b >= t ? b - t : 0);
}

template <typename T>
size_t UnboundedWSQ<T>::capacity() const noexcept {
  return _array.load(std::memory_order_relaxed)->capacity();
}

template <typename T>
void UnboundedWSQ<T>::push(T o) {
  int64_t b = _bottom.load(std::memory_order_relaxed);
  int64_t t = _top.load(std::memory_order_acquire);
  Array*  a = _array.load(std::memory_order_relaxed);

  if(a->capacity() < static_cast<size_t>(b - t + 1)) [[unlikely]] {
    a = _resize_array(a, b, t);
  }

  a->push(b, o);
  std::atomic_thread_fence(std::memory_order_release);
  _bottom.store(b + 1, std::memory_order_release);
}

template <typename T>
template <typename I>
void UnboundedWSQ<T>::bulk_push(I first, size_t N) {
  if(N == 0) return;

  int64_t b = _bottom.load(std::memory_order_relaxed);
  int64_t t = _top.load(std::memory_order_acquire);
  Array*  a = _array.load(std::memory_order_relaxed);

  if((b - t + N) > a->capacity()) [[unlikely]] {
    a = _resize_array(a, b, t, N);
  }

  for(size_t i=0; i<N; ++i) {
    a->push(b++, first[i]);
  }
  std::atomic_thread_fence(std::memory_order_release);
  _bottom.store(b, std::memory_order_release);
}

template <typename T>
typename UnboundedWSQ<T>::value_type UnboundedWSQ<T>::pop() {
  int64_t b = _bottom.load(std::memory_order_relaxed) - 1;
  Array*  a = _array.load(std::memory_order_relaxed);
  _bottom.store(b, std::memory_order_relaxed);
  std::atomic_thread_fence(std::memory_order_seq_cst);
  int64_t t = _top.load(std::memory_order_relaxed);

  auto item = empty_value();

  if(t <= b) {
    item = a->pop(b);
    if(t == b) {
      if(!_top.compare_exchange_strong(t, t+1,
            std::memory_order_seq_cst, std::memory_order_relaxed)) {
        item = empty_value();
      }
      _bottom.store(b + 1, std::memory_order_relaxed);
    }
  }
  else {
    _bottom.store(b + 1, std::memory_order_relaxed);
  }

  return item;
}

template <typename T>
typename UnboundedWSQ<T>::value_type UnboundedWSQ<T>::steal() {
  int64_t t = _top.load(std::memory_order_acquire);
  std::atomic_thread_fence(std::memory_order_seq_cst);
  int64_t b = _bottom.load(std::memory_order_acquire);

  auto item = empty_value();

  if(t < b) {
    Array* a = _array.load(std::memory_order_consume);
    item = a->pop(t);
    if(!_top.compare_exchange_strong(t, t+1,
          std::memory_order_seq_cst, std::memory_order_relaxed)) {
      return empty_value();
    }
  }

  return item;
}

template <typename T>
typename UnboundedWSQ<T>::value_type
UnboundedWSQ<T>::steal_with_feedback(size_t& num_empty_steals) {
  int64_t t = _top.load(std::memory_order_acquire);
  std::atomic_thread_fence(std::memory_order_seq_cst);
  int64_t b = _bottom.load(std::memory_order_acquire);

  auto item = empty_value();

  if(t < b) {
    num_empty_steals = 0;
    Array* a = _array.load(std::memory_order_consume);
    item = a->pop(t);
    if(!_top.compare_exchange_strong(t, t+1,
          std::memory_order_seq_cst, std::memory_order_relaxed)) {
      return empty_value();
    }
  }
  else {
    ++num_empty_steals;
  }

  return item;
}

template <typename T>
typename UnboundedWSQ<T>::Array*
UnboundedWSQ<T>::_resize_array(Array* a, int64_t b, int64_t t) {
  Array* tmp = a->resize(b, t);
  _garbage.push_back(a);
  _array.store(tmp, std::memory_order_release);
  return tmp;
}

template <typename T>
typename UnboundedWSQ<T>::Array*
UnboundedWSQ<T>::_resize_array(Array* a, int64_t b, int64_t t, size_t N) {
  Array* tmp = a->resize(b, t, N);
  _garbage.push_back(a);
  _array.store(tmp, std::memory_order_release);
  return tmp;
}

// ----------------------------------------------------------------------------
// BoundedWSQ
// ----------------------------------------------------------------------------

template <typename T, size_t LogSize = WSQ_DEFAULT_BOUNDED_LOG_SIZE>
class BoundedWSQ {

  static constexpr size_t BufferSize = size_t{1} << LogSize;
  static constexpr size_t BufferMask = BufferSize - 1;

  static_assert(BufferSize >= 2 && (BufferSize & BufferMask) == 0);

  // Original layout: _top, _bottom, _buffer each on their own cache lines
  alignas(WSQ_CACHELINE_SIZE) std::atomic<int64_t> _top    {0};
  alignas(WSQ_CACHELINE_SIZE) std::atomic<int64_t> _bottom {0};
  alignas(WSQ_CACHELINE_SIZE) std::atomic<T>        _buffer[BufferSize];

public:

  using value_type = std::conditional_t<std::is_pointer_v<T>, T, std::optional<T>>;

  BoundedWSQ()  = default;
  ~BoundedWSQ() = default;

  BoundedWSQ(const BoundedWSQ&)            = delete;
  BoundedWSQ& operator=(const BoundedWSQ&) = delete;

  bool   empty()    const noexcept;
  size_t size()     const noexcept;
  static constexpr size_t capacity() noexcept { return BufferSize; }

  template <typename O>
  bool try_push(O&& item);

  template <typename I>
  size_t try_bulk_push(I first, size_t N);

  value_type pop();
  value_type steal();
  value_type steal_with_feedback(size_t& num_empty_steals);

  static constexpr value_type empty_value() noexcept {
    if constexpr (std::is_pointer_v<T>) return T{nullptr};
    else                                return std::optional<T>{std::nullopt};
  }
};

// --- BoundedWSQ implementation ---

template <typename T, size_t LogSize>
bool BoundedWSQ<T, LogSize>::empty() const noexcept {
  int64_t t = _top.load(std::memory_order_relaxed);
  int64_t b = _bottom.load(std::memory_order_relaxed);
  return b <= t;
}

template <typename T, size_t LogSize>
size_t BoundedWSQ<T, LogSize>::size() const noexcept {
  int64_t t = _top.load(std::memory_order_relaxed);
  int64_t b = _bottom.load(std::memory_order_relaxed);
  return static_cast<size_t>(b >= t ? b - t : 0);
}

template <typename T, size_t LogSize>
template <typename O>
bool BoundedWSQ<T, LogSize>::try_push(O&& o) {
  int64_t b = _bottom.load(std::memory_order_relaxed);
  int64_t t = _top.load(std::memory_order_acquire);

  if(static_cast<size_t>(b - t + 1) > BufferSize) [[unlikely]] {
    return false;
  }

  _buffer[b & BufferMask].store(std::forward<O>(o), std::memory_order_relaxed);
  std::atomic_thread_fence(std::memory_order_release);
  _bottom.store(b + 1, std::memory_order_release);
  return true;
}

template <typename T, size_t LogSize>
template <typename I>
size_t BoundedWSQ<T, LogSize>::try_bulk_push(I first, size_t N) {
  if(N == 0) return 0;

  int64_t b = _bottom.load(std::memory_order_relaxed);
  int64_t t = _top.load(std::memory_order_acquire);

  size_t r = BufferSize - static_cast<size_t>(b - t);
  size_t n = std::min(N, r);

  if(n > 0) {
    for(size_t i=0; i<n; ++i) {
      _buffer[b++ & BufferMask].store(first[i], std::memory_order_relaxed);
    }
    std::atomic_thread_fence(std::memory_order_release);
    _bottom.store(b, std::memory_order_release);
  }

  return n;
}

template <typename T, size_t LogSize>
typename BoundedWSQ<T, LogSize>::value_type BoundedWSQ<T, LogSize>::pop() {
  int64_t b = _bottom.load(std::memory_order_relaxed) - 1;
  _bottom.store(b, std::memory_order_relaxed);
  std::atomic_thread_fence(std::memory_order_seq_cst);
  int64_t t = _top.load(std::memory_order_relaxed);

  auto item = empty_value();

  if(t <= b) {
    item = _buffer[b & BufferMask].load(std::memory_order_relaxed);
    if(t == b) {
      if(!_top.compare_exchange_strong(t, t+1,
            std::memory_order_seq_cst, std::memory_order_relaxed)) {
        item = empty_value();
      }
      _bottom.store(b + 1, std::memory_order_relaxed);
    }
  }
  else {
    _bottom.store(b + 1, std::memory_order_relaxed);
  }

  return item;
}

template <typename T, size_t LogSize>
typename BoundedWSQ<T, LogSize>::value_type BoundedWSQ<T, LogSize>::steal() {
  int64_t t = _top.load(std::memory_order_acquire);
  std::atomic_thread_fence(std::memory_order_seq_cst);
  int64_t b = _bottom.load(std::memory_order_acquire);

  auto item = empty_value();

  if(t < b) {
    item = _buffer[t & BufferMask].load(std::memory_order_relaxed);
    if(!_top.compare_exchange_strong(t, t+1,
          std::memory_order_seq_cst, std::memory_order_relaxed)) {
      return empty_value();
    }
  }

  return item;
}

template <typename T, size_t LogSize>
typename BoundedWSQ<T, LogSize>::value_type
BoundedWSQ<T, LogSize>::steal_with_feedback(size_t& num_empty_steals) {
  int64_t t = _top.load(std::memory_order_acquire);
  std::atomic_thread_fence(std::memory_order_seq_cst);
  int64_t b = _bottom.load(std::memory_order_acquire);

  auto item = empty_value();

  if(t < b) {
    num_empty_steals = 0;
    item = _buffer[t & BufferMask].load(std::memory_order_relaxed);
    if(!_top.compare_exchange_strong(t, t+1,
          std::memory_order_seq_cst, std::memory_order_relaxed)) {
      return empty_value();
    }
  }
  else {
    ++num_empty_steals;
  }

  return item;
}

}  // namespace wsq
