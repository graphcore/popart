// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_BASICOPTIONALS_HPP
#define GUARD_NEURALNET_BASICOPTIONALS_HPP

#include <ostream>
#include <popart/names.hpp>

namespace popart {

[[noreturn]] void noValueBasicOptionalError();

/**
 * A temporary solution to removing boost::optional from certain header files
 * This class is an incomplete replacement of boost::optional (and
 * std::optional).
 *
 * template parameter T: the type which will optionally be stored
 * template parameter V: has no effect, but enables compiler errors when two
 * objects of type T should not be compared
 *
 * */
template <typename T, uint32_t V = 0> class BasicOptional {
public:
  /**
   * Construct an unset BasicOptional<T>
   * */
  BasicOptional() noexcept : isSet(false) {}

  /**
   * Create a set BasicOptional<T> from a value
   * */
  BasicOptional(T t) : isSet(true), value(t) {}
  BasicOptional(const BasicOptional<T, V> &rhs) = default;

  BasicOptional<T, V> &operator=(const BasicOptional<T, V> &) = default;

  BasicOptional<T, V> &operator=(const T &t) {
    isSet = true;
    value = t;
    return *this;
  }

  /**
   * Get a constant reference to the value
   * */
  const T &operator*() const & {
    if (!isSet) {
      noValueBasicOptionalError();
    }
    return value;
  }

  /**
   * Get a reference to the value
   * */
  T &operator*() & {
    if (!isSet) {
      noValueBasicOptionalError();
    }
    return value;
  }

  /**
   * Return true if set. Can be used as:
   *
   * BasicOptional<Foo> foo(6);
   * if (foo){
   *   *foo = 7;
   * }
   * */
  explicit operator bool() const { return isSet; }

  void reset() noexcept { isSet = false; }

private:
  bool isSet;
  T value;
};

template <class T, uint32_t V>
bool operator==(const BasicOptional<T, V> &a, const BasicOptional<T, V> &b) {
  if (!a && !b) {
    return true;
  }
  return a && b && *a == *b;
}

template <class T, uint32_t V>
bool operator==(T a, const BasicOptional<T, V> &b) {
  return b && (*b == a);
}

template <class T, uint32_t V>
bool operator==(const BasicOptional<T, V> &a, T b) {
  return operator==(b, a);
}

template <class T, uint32_t V>
bool operator!=(const BasicOptional<T, V> &a, const BasicOptional<T, V> &b) {
  return !operator==(a, b);
}

template <class T, uint32_t V>
bool operator!=(const BasicOptional<T, V> &a, T b) {
  return !operator==(a, b);
}

template <class T, uint32_t V>
bool operator!=(T a, const BasicOptional<T, V> &b) {
  return !operator==(a, b);
}

using OptionalVGraphId             = BasicOptional<VGraphId, 2>;
using OptionalPipelineStage        = BasicOptional<PipelineStage, 3>;
using OptionalExecutionPhase       = BasicOptional<ExecutionPhase, 5>;
using OptionalBatchSerializedPhase = BasicOptional<BatchSerializedPhase, 7>;

template <typename T, uint32_t V>
std::ostream &operator<<(std::ostream &ost, const BasicOptional<T, V> &bo) {
  if (!bo) {
    ost << "none";
  } else {
    ost << *bo;
  }
  return ost;
}

} // namespace popart

#endif
