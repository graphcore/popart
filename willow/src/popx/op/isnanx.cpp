// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <popops/ElementWise.hpp>
#include <popart/error.hpp>
#include <popart/op/isnan.hpp>
#include <popart/popx/op/isnanx.hpp>
#include <popart/popx/opxmanager.hpp>

namespace popart {
namespace popx {

IsNaNx::IsNaNx(Op *op, Devicex *devicex) : ElementWiseUnaryOpx(op, devicex) {
  verifyOp<IsNaN>(op, Onnx::Operators::IsNaN_9);
}

void IsNaNx::grow(poplar::program::Sequence &prog) const {
  setOutTensor(IsNaN::getOutIndex(),
               popops::map(graph(),
                           popops::expr::BinaryOpType::NOT_EQUAL,
                           getInTensor(IsNaN::getInIndex()),
                           getInTensor(IsNaN::getInIndex()),
                           prog,
                           debugPrefix()));
}

namespace {
OpxCreator<IsNaNx> IsNaNxCreator(Onnx::Operators::IsNaN_9);
} // namespace

} // namespace popx
} // namespace popart
