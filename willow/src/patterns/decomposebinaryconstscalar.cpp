// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <map>
#include <string>
#include <utility>
#include <vector>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/binaryconstscalar.hpp>
#include <popart/patterns/decomposebinaryconstscalar.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorinfo.hpp>

#include "popart/datatype.hpp"
#include "popart/error.hpp"
#include "popart/half.hpp"
#include "popart/logging.hpp"
#include "popart/op.hpp"
#include "popart/operators.hpp"
#include "popart/patterns/patterns.hpp"
#include "popart/tensors.hpp"

namespace popart {

bool DecomposeBinaryConstScalar::matches(Op *op) const {
  return op->isConvertibleTo<BinaryConstScalarOp>();
}

namespace {
using OpType = decltype(Onnx::AiOnnx::OpSet9::Mul);
OpType convert(BinaryConstScalarOp::Type n) {

  static std::map<BinaryConstScalarOp::Type, OpType> M{
      {BinaryConstScalarOp::Type::Mul, Onnx::AiOnnx::OpSet9::Mul},
      {BinaryConstScalarOp::Type::Add, Onnx::AiOnnx::OpSet9::Add},
      {BinaryConstScalarOp::Type::Pow, Onnx::AiOnnx::OpSet9::Pow},
      {BinaryConstScalarOp::Type::Div, Onnx::AiOnnx::OpSet9::Div},
      {BinaryConstScalarOp::Type::Sub, Onnx::AiOnnx::OpSet9::Sub},
  };

  const auto found = M.find(n);
  if (found == M.cend()) {
    throw error("Unexpected Type in converting from BinaryConstScalarOp::Type "
                "to Onnx::AiOnnx type");
  }
  return found->second;
}
} // namespace

std::vector<const Tensor *> DecomposeBinaryConstScalar::touches(Op *) const {
  return {};
}

bool DecomposeBinaryConstScalar::apply(Op *op_) const {
  auto &op    = *dynamic_cast<BinaryConstScalarOp *>(op_);
  auto input  = op.inTensor(BinaryConstScalarOp::getInIndex());
  auto output = op.outTensor(BinaryConstScalarOp::getOutIndex());

  auto scalarId = op.getIr().createIntermediateTensorId("scalar");

  auto v_f32 = op.value();
  // Handle special case for halves.
  // TODO: T37480 Move this fix for halves down in the code.
  if (input->info.dataType() == DataType::FLOAT16) {
    std::vector<float16_t> data{v_f32};
    op.getGraph().getTensors().addConstInit(
        scalarId, {input->info.dataType(), {}}, data.data());
  } else {
    std::vector<float> data{v_f32};
    op.getGraph().getTensors().addConstInit(
        scalarId, {input->info.dataType(), {}}, data.data());
  }

  auto arg0 = scalarId;
  auto arg1 = input->id;
  if (op.scalarInIndex() == 1) {
    std::swap(arg0, arg1);
  }

  Op *binaryOp = makeReplacementOpInIr(convert(op.opType()), &op);

  // Remove the BinaryConstScalarOp
  op.disconnectAllInputs();
  op.disconnectAllOutputs();
  op.getGraph().eraseOp(op.id);

  binaryOp->connectInTensor(0, arg0);
  binaryOp->connectInTensor(1, arg1);
  binaryOp->connectOutTensor(0, output->id);
  binaryOp->setup();

  return true;
}

namespace {
static PatternCreator<popart::DecomposeBinaryConstScalar>
    binCaso("DecomposeBinaryConstScalar");

} // namespace

} // namespace popart
