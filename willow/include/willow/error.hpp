// Copyright (c) 2018, Graphcore Ltd, All rights reserved

#ifndef GUARD_NEURALNET_ERROR_HPP
#define GUARD_NEURALNET_ERROR_HPP

#include <stdexcept>
#include <string>

namespace willow {

// A  class for any exception which arises from the willow library.
class error : public std::runtime_error {
public:
  explicit error(const std::string &what) : std::runtime_error(who() + what) {}

private:
  std::string who() { return "willow: "; }
};

} // namespace willow

#endif
