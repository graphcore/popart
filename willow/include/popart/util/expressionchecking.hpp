// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_UTIL_EXPRESSIONCHECKING_HPP_
#define POPART_WILLOW_INCLUDE_POPART_UTIL_EXPRESSIONCHECKING_HPP_

#include <sstream>
#include <string>

#include "popart/logging.hpp"

namespace popart {
namespace internal {

/**
 * This class is used to store evaluated data in the `POPART_CHECK...` macros,
 * so that the left-hand side and right-hand side expressions don't get
 * evaluated multiple times.
 *
 * The expression can be calls to functions, for example `POPART_CHECK_EQ(foo(),
 * false)`. Evaluating an expression means calling `foo()`. This class is used
 * so we can only evaluate `foo()` once and store its result.
 */
template <typename T, typename U> class BinaryComparisonData {
private:
  const T &lhs_;
  const U &rhs_;
  // This stores the result of the `POPART_CHECK...`.
  bool result_;

public:
  BinaryComparisonData(const T &lhs, const U &rhs, bool result)
      : lhs_(lhs), rhs_(rhs), result_(result) {}

  const T &getLhs() const { return lhs_; }

  const U &getRhs() const { return rhs_; }

  explicit operator bool() const { return result_; }
};

/*
 * An instance of this class is constructed when a `POPART_CHECK...` fails. A
 * developer can append an error message to the instance of this class using the
 * `<<` operator.
 *
 * When this class gets destroyed, it throws an error with a message that
 * consists of the `FailedCheckThrower::prefix_`,
 * `FailedCheckThrower::extra_message_`, and `FailedCheckThrower::suffix_` in
 * that order.
 */
class FailedCheckThrower {
private:
  std::stringstream extra_message_;
  const std::string prefix_;
  const std::string suffix_;

  std::string buildErrorMessage() const;

public:
  FailedCheckThrower(const std::string prefix, const std::string suffix);

  FailedCheckThrower(const char *prefix, const char *suffix);

  ~FailedCheckThrower() noexcept(false);

  template <typename V> FailedCheckThrower &operator<<(const V &extra_message) {
    extra_message_ << extra_message;
    return *this;
  }
};

} // namespace internal
} // namespace popart

#define POPART_DEFINE_EVALUATE_BINARY_EXPRESSION_IMPL(name, op)                \
  template <typename T, typename U>                                            \
  inline ::popart::internal::BinaryComparisonData<T, U> evaluate##name(        \
      T &&lhs, U &&rhs) {                                                      \
    return ::popart::internal::BinaryComparisonData<T, U>(                     \
        std::forward<T>(lhs), std::forward<U>(rhs), !(lhs op rhs));            \
  }

POPART_DEFINE_EVALUATE_BINARY_EXPRESSION_IMPL(CheckEq, ==)
POPART_DEFINE_EVALUATE_BINARY_EXPRESSION_IMPL(CheckGe, >=)
POPART_DEFINE_EVALUATE_BINARY_EXPRESSION_IMPL(CheckGt, >)
POPART_DEFINE_EVALUATE_BINARY_EXPRESSION_IMPL(CheckLe, <=)
POPART_DEFINE_EVALUATE_BINARY_EXPRESSION_IMPL(CheckLt, <)
POPART_DEFINE_EVALUATE_BINARY_EXPRESSION_IMPL(CheckNe, !=)

#undef POPART_DEFINE_EVALUATE_BINARY_EXPRESSION_IMPL

#ifndef NDEBUG

// In debug mode, when a binary expression check fails, the file name and line
// number of the check are prepended to the message.
#define POPART_CHECK_BINARY_EXPRESSION(name, op, opposite_op, lhs, rhs)        \
  if (auto data____ = evaluate##name(lhs, rhs))                                \
  ::popart::internal::FailedCheckThrower(                                      \
      ::popart::logging::format("{}:{} Check {} {} {} has failed.",            \
                                __FILE__,                                      \
                                __LINE__,                                      \
                                #lhs,                                          \
                                op,                                            \
                                #rhs),                                         \
      ::popart::logging::format(                                               \
          "[{} {} {}]", data____.getLhs(), opposite_op, data____.getRhs()))

// In debug mode, when a unary expression check fails, the file name and line
// number of the check are prepended to the message.
#define POPART_CHECK_UNARY_EXPRESSION(expr)                                    \
  if (!(expr))                                                                 \
  ::popart::internal::FailedCheckThrower(                                      \
      ::popart::logging::format(                                               \
          "{}:{} Check {} has failed.", __FILE__, __LINE__, #expr),            \
      "")
#else

#define POPART_CHECK_BINARY_EXPRESSION(name, op, opposite_op, lhs, rhs)        \
  if (auto data____ = evaluate##name(lhs, rhs))                                \
  ::popart::internal::FailedCheckThrower(                                      \
      ::popart::logging::format("Check {} {} {} has failed.", #lhs, op, #rhs), \
      ::popart::logging::format(                                               \
          "[{} {} {}]", data____.getLhs(), opposite_op, data____.getRhs()))

#define POPART_CHECK_UNARY_EXPRESSION(expr)                                    \
  if (!(expr))                                                                 \
  ::popart::internal::FailedCheckThrower(                                      \
      ::popart::logging::format("Check {} has failed.", #expr), "")

#endif // NDEBUG

#define POPART_CHECK_EQ(lhs, rhs)                                              \
  POPART_CHECK_BINARY_EXPRESSION(CheckEq, "==", "!=", lhs, rhs)
#define POPART_CHECK_GE(lhs, rhs)                                              \
  POPART_CHECK_BINARY_EXPRESSION(CheckGe, ">=", "<", lhs, rhs)
#define POPART_CHECK_GT(lhs, rhs)                                              \
  POPART_CHECK_BINARY_EXPRESSION(CheckGt, ">", "<=", lhs, rhs)
#define POPART_CHECK_LE(lhs, rhs)                                              \
  POPART_CHECK_BINARY_EXPRESSION(CheckLe, "<=", ">", lhs, rhs)
#define POPART_CHECK_LT(lhs, rhs)                                              \
  POPART_CHECK_BINARY_EXPRESSION(CheckLt, "<", ">=", lhs, rhs)
#define POPART_CHECK_NE(lhs, rhs)                                              \
  POPART_CHECK_BINARY_EXPRESSION(CheckNe, "!=", "==", lhs, rhs)
#define POPART_CHECK(expr) POPART_CHECK_UNARY_EXPRESSION(expr)

#endif // POPART_WILLOW_INCLUDE_POPART_UTIL_EXPRESSIONCHECKING_HPP_
