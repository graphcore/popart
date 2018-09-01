// Copyright (c) 2018, Graphcore Ltd, All rights reserved

#ifndef GUARD_NEURALNET_ERROR_HPP
#define GUARD_NEURALNET_ERROR_HPP

#include <stdexcept>

namespace neuralnet {
// A  class for any exception which arises from the neuralnet library.
class error : public std::runtime_error {
public:
  error(const std::string &what) : std::runtime_error(what) {}
};


} // namespace neuralnet

#endif
