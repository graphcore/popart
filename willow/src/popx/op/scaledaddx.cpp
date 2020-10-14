// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/scaledadd.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/scaledaddx.hpp>
#include <popart/popx/opxmanager.hpp>

#include <popops/ScaledAdd.hpp>

namespace popart {
namespace popx {

ScaledAddOpx::ScaledAddOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<ScaledAddOp>(op,
                        {Onnx::AiGraphcore::OpSet1::ScaledAdd,
                         Onnx::CustomOperators::ScaledAddLhsInplace,
                         Onnx::CustomOperators::ScaledAddRhsInplace});
}

ScaledAddLhsInplaceOpx::ScaledAddLhsInplaceOpx(Op *op, Devicex *devicex)
    : ScaledAddOpx(op, devicex) {
  verifyOp<ScaledAddLhsInplaceOp>(op,
                                  {Onnx::CustomOperators::ScaledAddLhsInplace});
}

ScaledAddRhsInplaceOpx::ScaledAddRhsInplaceOpx(Op *op, Devicex *devicex)
    : ScaledAddOpx(op, devicex) {
  verifyOp<ScaledAddRhsInplaceOp>(op,
                                  {Onnx::CustomOperators::ScaledAddRhsInplace});
}

poplar::Tensor ScaledAddOpx::compute(poplar::program::Sequence &prog,
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
        graph(), in0, s0, in1, s1, prog, debugPrefix("t_t_t_t"));
  } else if (s0.valid() && !s1.valid()) {
    throw error("Unsupported tensor scale0 with non-tensor scale1.");
  } else if (hasInput(ScaledAddOp::getScale1InIndex())) {
    if (s0f != 1.0) {
      throw error("Unsupported scale0 {} with tensor scale1.", s0f);
    }
    popops::scaledAddTo(graph(), in0, in1, s1, prog, debugPrefix("t_1_t_t"));
  } else {
    popops::scaledAddTo(
        graph(), in0, s0f, in1, s1f, prog, debugPrefix("t_c_t_c"));
  }
  return in0;
}

void ScaledAddOpx::grow(poplar::program::Sequence &prog) const {
  auto &scaledAddOp = getOp<ScaledAddOp>();

  poplar::Tensor out;

  auto in0 = getInTensor(ScaledAddOp::getArg0InIndex());
  auto in1 = getInTensor(ScaledAddOp::getArg1InIndex());

  poplar::Tensor s0, s1;

  if (hasInput(ScaledAddOp::getScale0InIndex())) {
    s0 = getInTensor(ScaledAddOp::getScale0InIndex());
  }
  if (hasInput(ScaledAddOp::getScale1InIndex())) {
    s1 = getInTensor(ScaledAddOp::getScale1InIndex());
  }

  float s0f = scaledAddOp.getScale0();
  float s1f = scaledAddOp.getScale1();

  out = compute(prog, in0, in1, s0, s1, s0f, s1f, false);

  if (hasInViewChangers(ScaledAddOp::getArg0InIndex())) {
    setOutViewChangers(ScaledAddOp::getOutIndex(),
                       getInViewChangers(ScaledAddOp::getArg0InIndex()));
  }
  setOutTensor(ScaledAddOp::getOutIndex(), out);
}

void ScaledAddLhsInplaceOpx::grow(poplar::program::Sequence &prog) const {
  auto &scaledAddOp = getOp<ScaledAddLhsInplaceOp>();

  poplar::Tensor out;

  auto in0 = getInTensor(ScaledAddOp::getArg0InIndex());
  auto in1 = getInTensor(ScaledAddOp::getArg1InIndex());

  poplar::Tensor s0, s1;

  if (hasInput(ScaledAddOp::getScale0InIndex())) {
    s0 = getInTensor(ScaledAddOp::getScale0InIndex());
  }
  if (hasInput(ScaledAddOp::getScale1InIndex())) {
    s1 = getInTensor(ScaledAddOp::getScale1InIndex());
  }

  float s0f = scaledAddOp.getScale0();
  float s1f = scaledAddOp.getScale1();

  out = compute(prog, in0, in1, s0, s1, s0f, s1f, true);

  if (hasInViewChangers(ScaledAddOp::getArg0InIndex())) {
    setOutViewChangers(ScaledAddOp::getOutIndex(),
                       getInViewChangers(ScaledAddOp::getArg0InIndex()));
  }
  setOutTensor(ScaledAddOp::getOutIndex(), out);
}

void ScaledAddRhsInplaceOpx::grow(poplar::program::Sequence &prog) const {
  auto &scaledAddOp = getOp<ScaledAddRhsInplaceOp>();

  poplar::Tensor out;

  auto in0 = getInTensor(ScaledAddOp::getArg0InIndex());
  auto in1 = getInTensor(ScaledAddOp::getArg1InIndex());

  poplar::Tensor s0, s1;

  if (hasInput(ScaledAddOp::getScale0InIndex())) {
    s0 = getInTensor(ScaledAddOp::getScale0InIndex());
  }
  if (hasInput(ScaledAddOp::getScale1InIndex())) {
    s1 = getInTensor(ScaledAddOp::getScale1InIndex());
  }

  float s0f = scaledAddOp.getScale0();
  float s1f = scaledAddOp.getScale1();

  out = compute(prog, in1, in0, s1, s0, s1f, s0f, true);

  if (hasInViewChangers(ScaledAddOp::getArg0InIndex())) {
    setOutViewChangers(ScaledAddOp::getOutIndex(),
                       getInViewChangers(ScaledAddOp::getArg0InIndex()));
  }
  setOutTensor(ScaledAddOp::getOutIndex(), out);
}

namespace {
OpxCreator<ScaledAddOpx> scaledAddOpxCreator(Onnx::CustomOperators::ScaledAdd);
OpxCreator<ScaledAddLhsInplaceOpx>
    scaledAddLhsInplaceOpxCreator(Onnx::CustomOperators::ScaledAddLhsInplace);
OpxCreator<ScaledAddRhsInplaceOpx>
    scaledAddRhsInplaceOpxCreator(Onnx::CustomOperators::ScaledAddRhsInplace);

} // namespace
} // namespace popx
} // namespace popart
