// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <popops/ElementWise.hpp>
#include <popops/Reduce.hpp>
#include <popart/error.hpp>
#include <popart/ir.hpp>
#include <popart/op/identity.hpp>
#include <popart/optimizer.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/identityx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/tensor.hpp>

namespace popart {
namespace popx {

IdentityOpx::IdentityOpx(Op *op, Devicex *devicex)
    : ElementWiseUnaryOpx(op, devicex) {
  verifyOp<IdentityOp>(op, Onnx::Operators::Identity_1);
}

void IdentityOpx::grow(poplar::program::Sequence &prog) const {
  setOutTensor(0, Opx::cloneNcopy(prog, getInTensor(0)));
}

IdentityInplaceOpx::IdentityInplaceOpx(Op *op, Devicex *devicex)
    : Opx(op, devicex) {
  verifyOp<IdentityInplaceOp>(op);
}

void IdentityInplaceOpx::grow(poplar::program::Sequence &) const {
  setOutTensor(0, getInTensor(0));
}

IdentityGradOpx::IdentityGradOpx(Op *op, Devicex *devicex)
    : ElementWiseUnaryOpx(op, devicex) {
  verifyOp<IdentityGradOp>(op, Onnx::GradOperators::IdentityGrad);
}

void IdentityGradOpx::grow(poplar::program::Sequence &prog) const {
  setOutTensor(0, Opx::cloneNcopy(prog, getInTensor(0)));
}

IdentityLossOpx::IdentityLossOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<IdentityLossOp>(op, Onnx::CustomOperators::IdentityLoss);
}

void IdentityLossGradOpx::grow(poplar::program::Sequence &) const {
  IdentityLossGradOp &identitylossop = getOp<IdentityLossGradOp>();
  double scale;
  switch (identitylossop.identityl()->getReductionType()) {
  case ReductionType::Sum: {
    scale = 1;
    break;
  }
  case ReductionType::Mean: {
    uint64_t totalSamples =
        dv_p->getReplicationFactor() * getInTensor(0).numElements();
    scale = 1.0 / static_cast<double>(totalSamples);
    break;
  }
  default: {
    throw error("Unsupported reduction type for Loss {}", debugPrefix());
  }
  }
  poplar::Tensor ones = graph().addConstant(getInTensor(0).elementType(),
                                            getInTensor(0).shape(),
                                            scale,
                                            debugPrefix("ones"));
  graph().setTileMapping(ones, 0);
  setOutTensor(0, ones);
}

InputCreatorType IdentityLossOpx::getInputCreatorType(InIndex) const {
  return InputCreatorType::CanUnwind;
}

void IdentityLossOpx::grow(poplar::program::Sequence &prog) const {
  const IdentityLossOp &op         = getOp<IdentityLossOp>();
  const IdentityLoss *identityloss = op.identityl();

  const poplar::Tensor &inTensor(getInTensor(0));

  if (identityloss->getReductionType() == ReductionType::NoReduction) {
    throw error("This should have been replaced by an Identity op rather than"
                "an IdentityLoss op");
  } else {

    auto inTensor1D = inTensor.flatten();

    double scale;
    switch (identityloss->getReductionType()) {
    case ReductionType::Sum: {
      scale = 1.0;
      break;
    }
    case ReductionType::Mean: {
      double totalSamples = static_cast<double>(dv_p->getReplicationFactor()) *
                            static_cast<double>(inTensor1D.dim(0));
      scale = 1.0 / totalSamples;
      break;
    }
    default: {
      throw error("Unsupported reduction type for Loss {}", debugPrefix());
    }
    }

    // t_scale is always expected to be FLOAT, regardless of the input type
    // to the reduction
    auto t_scale = getConst(poplar::FLOAT, {}, scale, debugPrefix("scale"));

    auto reduction = popops::reduce(graph(),
                                    inTensor1D,
                                    {0},
                                    {popops::Operation::ADD, false, t_scale},
                                    prog,
                                    debugPrefix("add"));

    setOutTensor(0, reduction);
  }
}

IdentityLossGradOpx::IdentityLossGradOpx(Op *op, Devicex *devicex)
    : Opx(op, devicex) {
  verifyOp<IdentityLossGradOp>(op, Onnx::GradOperators::IdentityLossGrad);
}

namespace {
OpxCreator<IdentityOpx> identityOpxCreator(Onnx::Operators::Identity_1);
OpxCreator<IdentityInplaceOpx>
    identityInplaceOpxCreator(Onnx::CustomOperators::IdentityInplace);
OpxCreator<IdentityLossOpx>
    identityLossOpxCreator(Onnx::CustomOperators::IdentityLoss);
OpxCreator<IdentityGradOpx>
    identityGradOpxCreator(Onnx::GradOperators::IdentityGrad);
OpxCreator<IdentityLossGradOpx>
    identityLossGradOpxCreator(Onnx::GradOperators::IdentityLossGrad);
} // namespace

} // namespace popx
} // namespace popart
