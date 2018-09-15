#ifndef GUARD_NEURALNET_UTIL_HPP
#define GUARD_NEURALNET_UTIL_HPP

#include <string>

namespace neuralnet {

// turn into a string and pad if necessary to
// some minimum length
template <typename T> std::string padded(T in, int padSize) {
  std::stringstream ss;
  ss << in;
  std::string out = ss.str();
  if (out.size() < padSize) {
    out.resize(padSize, ' ');
  }
  return out;
}

} // namespace neuralnet

#endif
