// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <snap/Graph.hpp>
#include <snap/Program.hpp>
#include <snap/Tensor.hpp>
#include <snap/popops/ElementWise.hpp>
#include <vector>
#include <popops/Expr.hpp>
#include <popops/ExprOp.hpp>
#include <popops/ScaledAdd.hpp>
#include <popart/op/sgd1nesterov.hpp>
#include <popart/popx/op/sgd1nesterovx.hpp>
#include <popart/popx/opxmanager.hpp>

#include "popart/error.hpp"
#include "popart/graphcoreoperators.hpp"
#include "popart/logging.hpp"
#include "popart/operatoridentifier.hpp"
#include "popart/popx/popopx.hpp"

namespace pe = popops::expr;

namespace popart {
class Op;

namespace popx {
class Devicex;

SGD1NesterovOpx::SGD1NesterovOpx(Op *op, Devicex *devicex)
    : PopOpx(op, devicex) {
  verifyOp<SGD1NesterovOp>(op, {Onnx::CustomOperators::SGD1Nesterov});
}

// return s0 * in0 + s1 * in1.
snap::Tensor SGD1NesterovOpx::compute(snap::program::Sequence &prog,
                                      snap::Tensor in0,
                                      snap::Tensor in1,
                                      snap::Tensor s0,
                                      snap::Tensor s1,
                                      float s0f,
                                      float s1f,
                                      bool inplace) const {
  if (!inplace) {
    in0 = cloneNcopy(prog, in0);
  }

  if (s0.valid() && s1.valid()) {
    popops::scaledAddTo(graph().getPoplarGraph(),
                        in0.getPoplarTensor(),
                        s0.getPoplarTensor(),
                        in1.getPoplarTensor(),
                        s1.getPoplarTensor(),
                        prog.getPoplarSequence(),
                        debugContext("nonConstScaledAddS0S1"));
  } else if (s0.valid() && !s1.valid()) {
    snap::popops::mapInPlace(
        graph(),
        pe::Mul(pe::_1, pe::Cast(pe::_2, in0.elementType())),
        {in0, s0},
        prog,
        debugContext("nonConstScalingS0"));
    popops::scaledAddTo(
        graph().getPoplarGraph(),
        in0.getPoplarTensor(),
        in1.getPoplarTensor(),
        s1f,
        prog.getPoplarSequence(),
        debugContext("constScaledAddS1_" + std::to_string(s1f)));
  } else if (!s0.valid() && s1.valid()) {
    snap::popops::mapInPlace(
        graph(),
        pe::Mul(pe::_1, pe::Const(s0f)),
        {in0},
        prog,
        debugContext("constScalingS0_" + std::to_string(s0f)));
    popops::scaledAddTo(graph().getPoplarGraph(),
                        in0.getPoplarTensor(),
                        in1.getPoplarTensor(),
                        s1.getPoplarTensor(),
                        prog.getPoplarSequence(),
                        debugContext("nonConstScaledAddS1"));
  } else {
    popops::scaledAddTo(graph().getPoplarGraph(),
                        in0.getPoplarTensor(),
                        s0f,
                        in1.getPoplarTensor(),
                        s1f,
                        prog.getPoplarSequence(),
                        debugContext("constScaledAddS0_" + std::to_string(s0f) +
                                     "_S1_" + std::to_string(s1f)));
  }
  return in0;
}

void SGD1NesterovOpx::grow(snap::program::Sequence &prog) const {
  auto &op = getOp<SGD1NesterovOp>();

  auto grad     = getInTensor(SGD1NesterovOp::getGradInIndex());
  auto weight   = getInTensor(SGD1NesterovOp::getWeightInIndex());
  auto velocity = getInTensor(SGD1NesterovOp::getVelocityInIndex());
  snap::Tensor out;

  snap::Tensor ils, wd, ngsf, mm;

  if (hasInput(SGD1NesterovOp::getInverseLossScaleInIndex())) {
    ils = getInTensor(SGD1NesterovOp::getInverseLossScaleInIndex());
  }
  if (hasInput(SGD1NesterovOp::getWdInIndex())) {
    wd = getInTensor(SGD1NesterovOp::getWdInIndex());
  }
  if (hasInput(SGD1NesterovOp::getNgsfInIndex())) {
    ngsf = getInTensor(SGD1NesterovOp::getNgsfInIndex());
  }
  if (hasInput(SGD1NesterovOp::getMmInIndex())) {
    mm = getInTensor(SGD1NesterovOp::getMmInIndex());
  }

  float ilsf  = op.getInverseLossScale();
  float wdf   = op.getWd();
  float ngsff = op.getNgsf();
  float mmf   = op.getMm();

  out = compute(prog, grad, weight, ils, wd, ilsf, wdf, false);
  out = compute(prog, out, velocity, ngsf, mm, ngsff, mmf, true);

  if (hasInViewChangers(SGD1NesterovOp::getGradInIndex())) {
    setOutViewChangers(SGD1NesterovOp::getOutIndex(),
                       getInViewChangers(SGD1NesterovOp::getGradInIndex()));
  }
  setOutTensor(SGD1NesterovOp::getOutIndex(), out);
}

namespace {
OpxCreator<SGD1NesterovOpx>
    sgd1NesterovOpxCreator(Onnx::CustomOperators::SGD1Nesterov);
} // namespace
} // namespace popx
} // namespace popart
