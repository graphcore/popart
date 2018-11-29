#ifndef GUARD_NEURALNET_UTIL_HPP
#define GUARD_NEURALNET_UTIL_HPP

#include <memory>
#include <sstream>
#include <string>

namespace poponnx {

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

template <typename X, typename Y>
std::vector<Y> vXtoY(const std::vector<X> &c0) {
  std::vector<Y> c1;
  c1.reserve(c0.size());
  for (const X &v0 : c0) {
    c1.push_back(static_cast<Y>(v0));
  }
  return c1;
}

// TODO : If we move to C++14, this function will be standard.
template <typename T, typename... Args>
std::unique_ptr<T> make_unique(Args &&... args) {
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

namespace util {

/// Zip a pair of sequences with a given function into a third sequence.
///
/// \param begin1 end1  - The first sequence of elements.
/// \param begin2 end2  - The second sequence of elements.
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
/// \param begin1 end1  - The first sequence of elements.
/// \param begin2 end2  - The second sequence of elements.
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
} // namespace poponnx

#endif
