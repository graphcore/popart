// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <vector>
#include <popart/ces/identityce.hpp>
#include <popart/tensor.hpp>
#include <popart/tensordata.hpp>

#include "popart/ces/constexpr.hpp"
#include "popart/error.hpp"
#include "popart/logging.hpp"
#include "popart/tensorinfo.hpp"

namespace popart {
class Op;

ConstExprIdentity::ConstExprIdentity(Op *op_) : ConstExprOp(op_) {}

std::vector<char> ConstExprIdentity::compute() {
  if (inTensor(0)->info.nbytes() != outInfo0().nbytes()) {
    throw error("This is not what identity should be doing");
  }
  char *data  = reinterpret_cast<char *>(inTensor(0)->tensorData()->data());
  auto nbytes = outInfo0().nbytes();
  return std::vector<char>(data, data + nbytes);
}

} // namespace popart
