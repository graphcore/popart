#ifndef GUARD_NEURALNET_UTIL_HPP
#define GUARD_NEURALNET_UTIL_HPP

#include <sstream>
#include <string>

namespace willow {

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

template <class T> void appendSequence(std::stringstream &ss, const T &t) {
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

} // namespace willow

#endif
