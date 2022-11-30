#ifndef GUARD_NEURALNET_ANY_HPP
#define GUARD_NEURALNET_ANY_HPP

#include "popart/vendored/anylite.hpp"
#include <utility>

#define REQUIRES_T(...) , typename std::enable_if<(__VA_ARGS__), int>::type = 0

namespace popart {

/**
 * @brief Wrapper around nonstd::any.
 *
 * In order to be able to access nonstd::any as popart::any, we have to wrap it
 * in a class that has identical members except they call the same method of
 * nonstd::any in their implementation. This approach is taken because the more
 * straightforward solution of `using any = nonstd::any` actually raises
 * compiler errors because the declaration here conflicts with forward
 * declarations of popart::any that appear elsewhere in the codebase.
 */
class any {
public:
  constexpr any() noexcept : _impl() {}

  any(any const &other) : _impl(other.get_any()) {}

  any(any &&other) : _impl(std::forward<nonstd::any>(other.get_any())) {}

  template <class ValueType,
            class T = typename std::decay<ValueType>::type REQUIRES_T(
                !std::is_same<T, any>::value)>
  any(ValueType &&value) noexcept : _impl(std::forward<ValueType>(value)) {}

  template <
      class T,
      class... Args REQUIRES_T(std::is_constructible<T, Args &&...>::value)>
  explicit any(nonstd_lite_in_place_type_t(T), Args &&...args)
      : _impl(std::forward<Args>(args)...) {}

  template <
      class T,
      class U,
      class... Args REQUIRES_T(std::is_constructible<T,
                                                     std::initializer_list<U> &,
                                                     Args &&...>::value)>
  explicit any(nonstd_lite_in_place_type_t(T),
               std::initializer_list<U> il,
               Args &&...args)
      : _impl(il, std::forward<Args>(args)...) {}

  ~any() = default;

  any &operator=(any const &other) {
    any(other).swap(*this);
    return *this;
  }

  any &operator=(any &&other) noexcept {
    any(std::move(other)).swap(*this);
    return *this;
  }

  template <class ValueType,
            class T = typename std::decay<ValueType>::type REQUIRES_T(
                !std::is_same<T, any>::value)>
  any &operator=(T &&value) {
    any(std::move(value)).swap(*this);
    return *this;
  }

  void reset() noexcept { _impl.reset(); }

  void swap(any &other) noexcept { other.get_any().swap(_impl); }

  bool has_value() const noexcept { return _impl.has_value(); }

  const std::type_info &type() const noexcept { return _impl.type(); }

  nonstd::any &get_any() noexcept { return _impl; }
  const nonstd::any &get_any() const noexcept { return _impl; }

private:
  nonstd::any _impl;
};

// Wrappers around any_cast overloads. Because these functions are templated
// in anylite, we can't just inject them into the popart namespace using
// `using`. Moreover, we need these wrappers to retrieve `nonstd::any`
// from `popart::any`.

template <class V>
inline auto any_cast(any &a) -> decltype(nonstd::any_cast<V>(a)) {
  return nonstd::any_cast<V>(a.get_any());
}

template <class V>
inline auto any_cast(any const &a) -> decltype(nonstd::any_cast<V>(a)) {
  return nonstd::any_cast<V>(a.get_any());
}

template <class V>
inline auto any_cast(any &&a) -> decltype(nonstd::any_cast<V>(a)) {
  return nonstd::any_cast<V>(a.get_any());
}
} // namespace popart

#endif // GUARD_NEURALNET_ANY_HPP
