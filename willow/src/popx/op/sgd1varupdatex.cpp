// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <poplar/Tensor.hpp>
#include <popops/ElementWise.hpp>
#include <popops/ScaledAdd.hpp>
#include <popart/op/sgd1varupdate.hpp>
#include <popart/op/varupdate.hpp>
#include <popart/optimizervalue.hpp>
#include <popart/popx/op/sgd1varupdatex.hpp>
#include <popart/popx/opxmanager.hpp>

#include "popart/graphcoreoperators.hpp"
#include "popart/popx/op/varupdatex.hpp"

namespace poplar {
namespace program {
class Sequence;
} // namespace program
} // namespace poplar

namespace popart {
class Op;

namespace popx {
class Devicex;

SGD1VarUpdateOpx::SGD1VarUpdateOpx(Op *op, Devicex *devicex)
    : VarUpdateOpx(op, devicex) {
  verifyOp<SGD1VarUpdateOp>(op, Onnx::CustomOperators::SGD1VarUpdate);
}

void SGD1VarUpdateOpx::grow(poplar::program::Sequence &prog) const {

  // see optimizer.hpp for the equations implemented here

  auto sgd1varUpdateOp = getOp<SGD1VarUpdateOp>();

  poplar::Tensor velocity =
      getInTensor(VarUpdateWithUpdaterOp::getUpdaterInIndex());

  poplar::Tensor weights = getInTensor(VarUpdateOp::getVarToUpdateInIndex());

  // non-const scaled learning rate case
  if (!sgd1varUpdateOp.initSlr1.isConst()) {
    popops::scaledAddTo(
        graph(),
        weights,
        velocity,
        popops::neg(graph(),
                    getInTensor(SGD1VarUpdateOp::getSlr1InIndex()),
                    prog,
                    debugContext("neg")),
        prog,
        debugContext("nonConstScaledSubtractSGD1"));
  }

  // const scaled learning rate case
  else {
    popops::scaledAddTo(graph(),
                        weights,
                        velocity,
                        -sgd1varUpdateOp.initSlr1.val(),
                        prog,
                        debugContext("constScaledSubtractSGD1"));
  }

  if (hasInViewChangers(SGD1VarUpdateOp::getVarToUpdateInIndex())) {
    setOutViewChangers(
        SGD1VarUpdateOp::getUpdatedVarOutIndex(),
        getInViewChangers(SGD1VarUpdateOp::getVarToUpdateInIndex()));
  }
  // output is a reference to the updated input
  setOutTensor(SGD1VarUpdateOp::getUpdatedVarOutIndex(),
               getInTensor(SGD1VarUpdateOp::getVarToUpdateInIndex()));
}

namespace {
OpxCreator<SGD1VarUpdateOpx>
    sgd1VarUpdateOpxCreator(Onnx::CustomOperators::SGD1VarUpdate);
}
} // namespace popx
} // namespace popart
