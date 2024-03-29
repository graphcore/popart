// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <cstddef>
#include <ext/new_allocator.h>
#include <vector>
#include <poplar/Program.hpp>
#include <poplar/Tensor.hpp>
#include <poplar/Type.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Expr.hpp>
#include <popops/ExprOp.hpp>
#include <popops/OperationDef.hpp>
#include <popops/Reduce.hpp>
#include <popops/Zero.hpp>
#include <popart/error.hpp>
#include <popart/op/identity.hpp>
#include <popart/popx/op/identityx.hpp>
#include <popart/popx/opxmanager.hpp>

#include "popart/logging.hpp"
#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/operators.hpp"
#include "popart/popx/debugcontextx.hpp"
#include "popart/popx/op/elementwisex.hpp"
#include "popart/popx/opx.hpp"

namespace pe = popops::expr;

namespace popart {
namespace view {
class Region;
} // namespace view

namespace popx {
class Devicex;

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

void IdentityLossGradOpx::grow(poplar::program::Sequence &prog) const {
  IdentityLossGradOp &identitylossop = getOp<IdentityLossGradOp>();
  auto output                        = getInTensor(0);
  auto reference                     = getOutTensor(0);
  if (identitylossop.getReductionType() == ReductionType::NoReduction) {
    // Same as IdentityGradOpx
    prog.add(poplar::program::Copy(
        output, reference, false, debugContext("copy_identity")));
  } else {
    if (identitylossop.getReductionType() == ReductionType::Mean) {
      // Divide broadcasted tensor by total number of samples
      float scale = static_cast<float>(reference.numElements());

      output = popops::map(graph(),
                           pe::Divide(pe::_1, pe::Const(scale)),
                           {getInTensor(0)},
                           prog,
                           debugContext("div"));
    } else if (identitylossop.getReductionType() != ReductionType::Sum) {
      // Only mean and sum are supported.
      throw error("Unsupported reduction type for Loss {}",
                  debugContext().getPathName());
    }
    popops::zero(graph(),
                 reference,
                 prog,
                 debugContext("zero_identity_reference_tensor"));
    popops::addInPlace(graph(),
                       reference,
                       output,
                       prog,
                       debugContext("add_gradient_to_reference"));
  }
}

InputCreatorType IdentityLossOpx::getInputCreatorType(InIndex index) const {
  if (getOp<IdentityLossOp>().getReductionType() ==
          ReductionType::NoReduction &&
      index == IdentityLossOp::getInIndex()) {
    return InputCreatorType::CanUnwind;
  }
  return InputCreatorType::Deadend;
}

view::RegMap IdentityLossOpx::unwindRegion(InIndex, OutIndex) const {
  return [](const view::Region &r) { return view::Regions(1, r); };
}

poplar::Tensor IdentityLossOpx::unwindTensorLayout(poplar::Tensor tensor,
                                                   InIndex,
                                                   OutIndex) const {
  return tensor;
}

void IdentityLossOpx::grow(poplar::program::Sequence &prog) const {
  const IdentityLossOp &op = getOp<IdentityLossOp>();
  const auto &inTensor(getInTensor(0));

  if (op.getReductionType() == ReductionType::NoReduction) {
    setOutTensor(0, Opx::cloneNcopy(prog, inTensor));
  } else {

    auto inTensor1D = inTensor.flatten();

    double scale;
    switch (op.getReductionType()) {
    case ReductionType::Sum: {
      scale = 1.0;
      break;
    }
    case ReductionType::Mean: {
      double totalSamples = static_cast<double>(inTensor1D.dim(0));
      scale               = 1.0 / totalSamples;
      break;
    }
    // Making it explicit which data types we're not handling. Note that
    // the logic will fall through to the error.
    case ReductionType::NoReduction:
    default: {
      throw error("Unsupported reduction type for Loss {}",
                  debugContext().getPathName());
    }
    }

    // t_scale is always expected to be FLOAT, regardless of the input type
    // to the reduction
    auto t_scale = getConst(poplar::FLOAT, {}, scale, "scale");

    auto reduction = popops::reduce(graph(),
                                    inTensor1D,
                                    {0},
                                    {popops::Operation::ADD, false, t_scale},
                                    prog,
                                    debugContext("add"));

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
