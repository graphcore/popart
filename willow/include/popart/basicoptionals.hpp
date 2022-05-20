// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_BASICOPTIONALS_HPP
#define GUARD_NEURALNET_BASICOPTIONALS_HPP

#include <cstddef>
#include <cstdint>
#include <ostream>
#include <string>
#include <popart/graphid.hpp>
#include <popart/names.hpp>
#include <popart/tensorlocation.hpp> // IWYU pragma: keep

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

protected:
  bool isSet;
  T value;
};

/**
 * Template specialisation to get around an issue with deleted default
 * constructor of GraphId.
 */
class OptionalGraphId : public BasicOptional<GraphId, 11> {
public:
  using BasicOptional::BasicOptional;

  /**
   * Construct a new Optional Graph Id object
   *
   * Can't use BasicOptional() as this will try to call deleted GraphId(),
   * instead use empty string in GraphId(""), but set isSet false.
   */
  OptionalGraphId() : BasicOptional(GraphId("")) { isSet = false; };

  OptionalGraphId &operator=(const OptionalGraphId &) = default;

  /**
   * Construct a new Optional Graph Id object
   *
   * See :Implicit declaration of copy functions [depr.impldec]
   * The implicit definition of a copy constructor as defaulted is deprecated if
   * the class has a user-declared copy assignment operator or a user-declared
   * destructor. The implicit definition of a copy assignment operator as
   * defaulted is deprecated if the class has a user-declared copy constructor
   * or a user-declared destructor (15.4, 15.8). In a future revision of this
   * International Standard, these implicit definitions could become deleted
   * (11.4).
   */
  OptionalGraphId(OptionalGraphId &&)      = default;
  OptionalGraphId(const OptionalGraphId &) = default;
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
using OptionalTensorLocation       = BasicOptional<TensorLocation, 9>;
using OptionalStochasticRoundingMethod =
    BasicOptional<StochasticRoundingMethod, 10>;
// using OptionalGraphId = BasicOptional<GraphId, 11>; // See above

using OptionalCodeMemoryType = BasicOptional<CodeMemoryType, 12>;

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
