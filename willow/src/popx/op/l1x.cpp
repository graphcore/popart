// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <numeric>
#include <popart/error.hpp>
#include <popart/ir.hpp>
#include <popart/op/l1.hpp>
#include <popart/optimizer.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/l1x.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/tensor.hpp>

#include <popops/ElementWise.hpp>
#include <popops/Expr.hpp>
#include <popops/Reduce.hpp>

namespace pe = popops::expr;

namespace popart {
namespace popx {

L1Opx::L1Opx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<L1Op>(op, Onnx::CustomOperators::L1);
}

void L1GradOpx::grow(poplar::program::Sequence &prog) const {
  L1GradOp &l1gradop = getOp<L1GradOp>();

  double lambda = static_cast<double>(l1gradop.getLambda());

  // Signum : +1 of positive, -1 if negative, 0 if zero.
  poplar::Tensor signumTensor =
      popops::map(graph(),
                  popops::expr::UnaryOpType::SIGNUM,
                  getInTensor(L1GradOp::getFwdActInIndex()),
                  prog,
                  debugPrefix("Signum"));

  double scale = lambda;
  switch (l1gradop.getReductionType()) {
  case ReductionType::NoReduction:
    break;
  case ReductionType::Sum:
    break;
  case ReductionType::Mean: {
    double totalSamples = static_cast<double>(dv_p->getReplicationFactor()) *
                          static_cast<double>(getInTensor(0).numElements());
    scale = lambda / totalSamples;
    break;
  }
  default:
    throw error("Unsupported reduction type for Loss {}", debugPrefix());
  }

  auto t_scale =
      getConst(getInTensor(0).elementType(), {}, scale, debugPrefix("scale"));

  // scale the signum tensor:
  // - first by 'scale',  so +scale if positive, -scale if negative, 0 if zero
  // - by loss scaling factor
  // - then by input gradient

  auto gradTensor = popops::map(graph(),
                                pe::Mul(pe::_1, pe::_2),
                                {signumTensor, t_scale},
                                prog,
                                debugPrefix("multiply"));

  auto gradIn = getInTensor(L1GradOp::getGradInIndex());
  popops::mapInPlace(graph(),
                     pe::Mul(pe::_1, pe::_2),
                     {gradTensor, gradIn},
                     prog,
                     debugPrefix("scaledGradIn"));

  setOutTensor(0, gradTensor);
}

InputCreatorType L1Opx::getInputCreatorType(InIndex) const {
  return InputCreatorType::CanUnwind;
}

// lambda * sum_{0,..rank-1} |v|
void L1Opx::grow(poplar::program::Sequence &prog) const {
  const L1Op &l1op         = getOp<L1Op>();
  poplar::Tensor absTensor = popops::map(graph(),
                                         popops::expr::UnaryOpType::ABSOLUTE,
                                         getInTensor(0),
                                         prog,
                                         debugPrefix("abs"));

  if (absTensor.rank() == 0) {
    throw error("invalid tensor (rank-0) in L1Opx");
  }

  double lambda = static_cast<double>(l1op.getLambda());

  if (l1op.getReductionType() == ReductionType::NoReduction) {
    auto t_scale =
        getConst(absTensor.elementType(), {}, lambda, debugPrefix("scale"));

    auto scaled = popops::map(graph(),
                              popops::expr::BinaryOpType::MULTIPLY,
                              absTensor,
                              t_scale,
                              prog,
                              debugPrefix("add"));
    setOutTensor(0, scaled);
  } else {
    auto absTensor1D = absTensor.flatten();
    double scale     = lambda;

    switch (l1op.getReductionType()) {
    case ReductionType::Sum: {
      break;
    }
    case ReductionType::Mean: {
      double totalSamples = static_cast<double>(absTensor1D.dim(0));
      scale               = lambda / totalSamples;
      break;
    }
    // Making it explicit which data types we're not handling. Note that
    // the logic will fall through to the error.
    case ReductionType::NoReduction:
    default: {
      throw error("Unsupported reduction type for Loss {}", debugPrefix());
    }
    }

    // t_scale is always expected to be FLOAT, regardless of the input type
    // to the reduction
    auto t_scale   = getConst(poplar::FLOAT, {}, scale, debugPrefix("scale"));
    auto reduction = popops::reduce(graph(),
                                    absTensor1D,
                                    {0},
                                    {popops::Operation::ADD, false, t_scale},
                                    prog,
                                    debugPrefix("add"));
    setOutTensor(0, reduction);
  }
}

L1GradOpx::L1GradOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<L1GradOp>(op, Onnx::CustomGradOperators::L1Grad);
}

namespace {
OpxCreator<L1Opx> l1OpxCreator(Onnx::CustomOperators::L1);
OpxCreator<L1GradOpx> l1GradOpxCreator(Onnx::CustomGradOperators::L1Grad);
} // namespace

} // namespace popx
} // namespace popart
