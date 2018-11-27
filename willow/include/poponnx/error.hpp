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
  explicit error(const std::string &what);
};

enum ErrorSource {
  poponnx = 0,
  poplar  = 1,
  poplibs = 2,
  unknown = 3,
};

ErrorSource getErrorSource(const std::exception &e);

} // namespace willow

#endif
