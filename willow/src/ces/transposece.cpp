// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <cstring>
#include <memory>
#include <vector>
#include <poprithms/ndarray/dtype.hpp>
#include <poprithmshosttensor.hpp>
#include <popart/ces/transposece.hpp>
#include <popart/ndarraywrapper.hpp>
#include <popart/op/transpose.hpp>
#include <popart/tensor.hpp>

namespace popart {

ConstExprTranspose::ConstExprTranspose(Op *op_) : ConstExprOp(op_) {}

std::vector<char> ConstExprTranspose::compute() {

  auto perm = getOp<TransposeOp>().getPerm();

  if (perm.empty()) {
    // Default is to reverse the input shape
    for (int64_t i = inTensor(0)->info.rank() - 1; i >= 0; i--) {
      perm.push_back(i);
    }
  }

  std::vector<uint64_t> perm_u64;
  perm_u64.reserve(perm.size());
  for (auto d : perm) {
    perm_u64.push_back(static_cast<uint64_t>(d));
  }

  return getPoprithmsComputeHostTensor(*inTensor(0))
      .dimShuffle(perm_u64)
      .getNativeCharVector();
}

} // namespace popart
