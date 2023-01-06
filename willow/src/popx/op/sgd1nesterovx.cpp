// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <string>
#include <poplar/Tensor.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Expr.hpp>
#include <popops/ExprOp.hpp>
#include <popops/ScaledAdd.hpp>
#include <popart/op/sgd1nesterov.hpp>
#include <popart/popx/op/sgd1nesterovx.hpp>
#include <popart/popx/opxmanager.hpp>

#include "popart/graphcoreoperators.hpp"
#include "popart/popx/opx.hpp"

namespace poplar {
namespace program {
class Sequence;
} // namespace program
} // namespace poplar

namespace pe = popops::expr;

namespace popart {
class Op;

namespace popx {
class Devicex;

SGD1NesterovOpx::SGD1NesterovOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<SGD1NesterovOp>(op, {Onnx::CustomOperators::SGD1Nesterov});
}

// return s0 * in0 + s1 * in1.
poplar::Tensor SGD1NesterovOpx::compute(poplar::program::Sequence &prog,
                                        poplar::Tensor in0,
                                        poplar::Tensor in1,
                                        poplar::Tensor s0,
                                        poplar::Tensor s1,
                                        float s0f,
                                        float s1f,
                                        bool inplace) const {
  if (!inplace) {
    in0 = cloneNcopy(prog, in0);
  }

  if (s0.valid() && s1.valid()) {
    popops::scaledAddTo(
        graph(), in0, s0, in1, s1, prog, debugContext("nonConstScaledAddS0S1"));
  } else if (s0.valid() && !s1.valid()) {
    popops::mapInPlace(graph(),
                       pe::Mul(pe::_1, pe::Cast(pe::_2, in0.elementType())),
                       {in0, s0},
                       prog,
                       debugContext("nonConstScalingS0"));
    popops::scaledAddTo(
        graph(),
        in0,
        in1,
        s1f,
        prog,
        debugContext("constScaledAddS1_" + std::to_string(s1f)));
  } else if (!s0.valid() && s1.valid()) {
    popops::mapInPlace(graph(),
                       pe::Mul(pe::_1, pe::Const(s0f)),
                       {in0},
                       prog,
                       debugContext("constScalingS0_" + std::to_string(s0f)));
    popops::scaledAddTo(
        graph(), in0, in1, s1, prog, debugContext("nonConstScaledAddS1"));
  } else {
    popops::scaledAddTo(graph(),
                        in0,
                        s0f,
                        in1,
                        s1f,
                        prog,
                        debugContext("constScaledAddS0_" + std::to_string(s0f) +
                                     "_S1_" + std::to_string(s1f)));
  }
  return in0;
}

void SGD1NesterovOpx::grow(poplar::program::Sequence &prog) const {
  auto &op = getOp<SGD1NesterovOp>();

  auto grad     = getInTensor(SGD1NesterovOp::getGradInIndex());
  auto weight   = getInTensor(SGD1NesterovOp::getWeightInIndex());
  auto velocity = getInTensor(SGD1NesterovOp::getVelocityInIndex());
  poplar::Tensor out;

  poplar::Tensor ils, wd, ngsf, mm;

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
