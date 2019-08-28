#ifndef GUARD_NEURALNET_UTIL_HPP
#define GUARD_NEURALNET_UTIL_HPP

#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace popart {

char *getPopartEnvVar(std::string env_var);

// turn input into a string, and pads
// it if necessary to some minimum length `padSize'
template <typename T> std::string padded(T in, int padSize) {
  std::stringstream ss;
  ss << in;
  std::string out = ss.str();
  if (out.size() < padSize) {
    out.resize(padSize, ' ');
  }
  return out;
}

template <class T> void appendSequence(std::ostream &ss, const T &t) {
  int index = 0;
  ss << '[';
  for (auto &x : t) {
    if (index != 0) {
      ss << ' ';
    }
    ss << x;
    ++index;
  }
  ss << ']';
}

std::ostream &operator<<(std::ostream &ss, const std::vector<std::size_t> &v);

template <typename X, typename Y>
std::vector<Y> vXtoY(const std::vector<X> &c0) {
  std::vector<Y> c1;
  c1.reserve(c0.size());
  for (const X &v0 : c0) {
    c1.push_back(static_cast<Y>(v0));
  }
  return c1;
}

// Note: Template order different than for vXtoY!
template <typename T1, typename T2>
std::vector<T1> vector_cast(const std::vector<T2> &xs) {
  std::vector<T1> ys;

  ys.reserve(xs.size());
  for (const auto &x : xs) {
    ys.emplace_back(static_cast<T1>(x));
  }

  return ys;
}

template <typename Y> std::vector<Y> vBooltoY(const std::vector<bool> &c0) {
  std::vector<Y> c1;
  c1.reserve(c0.size());
  for (const bool v0 : c0) {
    c1.push_back(static_cast<Y>(v0));
  }
  return c1;
}

// Handy erase_if for std::map. Will be added to C++ 20 eventually
// https://en.cppreference.com/w/cpp/experimental/map/erase_if
template <class Key, class T, class Compare, class Alloc, class Pred>
void erase_if(std::map<Key, T, Compare, Alloc> &c, const Pred &pred) {
  for (auto it = c.begin(); it != c.end();) {
    if (pred(*it))
      it = c.erase(it);
    else
      ++it;
  }
}

namespace util {

/// Zip a pair of sequences with a given function into a third sequence.
///
/// \param begin1       - The start of the first sequence of elements.
/// \param end1         - The end of the first sequence of elements.
/// \param begin2       - The start of the second sequence of elements.
/// \param end2         - The end of the second sequence of elements.
/// \param obegin       - The beginning of the output sequence.
/// \param f            - The binary operation function object that will be
///                       applied. This function takes one value from each input
///                       sequence and produces a new value.
///
/// \note The IIter1 and IIter2 types must satisfy the `InputIterator` concept.
/// \note The OIter type must satisfy the `OutputIterator` concept.
/// \note The BinaryOperation must be a callable object with a signature of
///       `OIter::value_type f(IIter1::value_type, IIter2::value_type)`.
/// \note The output sequence must be at least as long as the shortest input
///       sequence.
template <typename IIter1,
          typename IIter2,
          typename OIter,
          typename BinaryOperation>
void zipWith(IIter1 begin1,
             IIter1 end1,
             IIter2 begin2,
             IIter2 end2,
             OIter obegin,
             BinaryOperation f) {
  while (begin1 != end1 && begin2 != end2) {
    *obegin = f(*begin1, *begin2);

    ++obegin;
    ++begin1;
    ++begin2;
  }
}

/// Count the number of differences between a pair of sequences.
///
/// \param begin1 - The start of the first sequence of elements.
/// \param end1   - The end of the first sequence of elements.
/// \param begin2 - The start of the second sequence of elements.
/// \param end2   - The end of the second sequence of elements.
///
/// \note The IIter1 and IIter2 types must satisfy the `InputIterator` concept.
template <typename IIter1, typename IIter2>
std::size_t
count_mismatch(IIter1 begin1, IIter1 end1, IIter2 begin2, IIter2 end2) {
  std::size_t result = 0;

  while (begin1 != end1 && begin2 != end2) {
    if (*begin1 != *begin2) {
      result++;
    }
    begin1++;
    begin2++;
  }

  return result + std::distance(begin1, end1) + std::distance(begin2, end2);
}

} // namespace util
} // namespace popart

// As per https://github.com/gabime/spdlog/issues/39 the operator<< needs to
// defined either before you include spdlog.h or placed in the srd namespace
namespace std {
template <typename T>
std::ostream &operator<<(std::ostream &ss, const std::vector<T> &v) {
  popart::appendSequence(ss, v);
  return ss;
}

template <typename T>
std::ostream &operator<<(std::ostream &ss, const std::set<T> &v) {
  popart::appendSequence(ss, v);
  return ss;
}

template <typename Key, typename Value>
std::ostream &operator<<(std::ostream &ss, const std::map<Key, Value> &v) {
  ss << "[";

  int comma_counter = 0;
  for (auto &key_value : v) {
    auto &key   = key_value.first;
    auto &value = key_value.second;
    ss << key << ": " << value;
    if (comma_counter < v.size()) {
      ss << ", ";
      comma_counter++;
    }
  }

  ss << "]";

  return ss;
}
} // namespace std

#endif
