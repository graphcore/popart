// Copyright (c) 2018, Graphcore Ltd, All rights reserved

#ifndef GUARD_NEURALNET_ERROR_HPP
#define GUARD_NEURALNET_ERROR_HPP

#include <stdexcept>
#include <string>

namespace willow {

/**
 * Exception class for poponnx
 */
class error : public std::runtime_error {
public:
  explicit error(const std::string &what) : std::runtime_error(what) {}
};

} // namespace willow

#endif
