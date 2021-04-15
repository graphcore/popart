// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <memory>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/add.hpp>
#include <popart/op/gemm.hpp>
#include <popart/op/matmul.hpp>
#include <popart/op/scale.hpp>
#include <popart/op/transpose.hpp>
#include <popart/patterns/gemmdecompositionpattern.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorinfo.hpp>

namespace popart {

bool GemmDecompositionPattern::matches(Op *op) const {
  return op->isConvertibleTo<GemmOp>();
}

std::vector<const Tensor *> GemmDecompositionPattern::touches(Op *) const {
  return {};
}

// output = add(scale(matmul(in_a, in_b), alpha), scale(in_c, beta))
bool GemmDecompositionPattern::apply(Op *op) const {
  auto output = op->outTensor(GemmOp::getOutIndex());
  op->disconnectOutTensor(output);

  // we assume this dynamic_cast call has been confirmed
  // to be valid via a previous call to GemmDecompositionPattern::matches
  auto gemm_op = dynamic_cast<GemmOp *>(op);

  auto &ir = op->getIr();

  // Get the first two inputs and transpose if necessary.
  auto in_a = op->inTensor(GemmOp::getAInIndex());
  op->disconnectInTensor(in_a);
  auto A = in_a->id;
  if (gemm_op->getTransA()) {
    auto tA = ir.createIntermediateTensorId(A);
    transposeTensor(A, tA, op);
    A = tA;
  }

  auto in_b = op->inTensor(GemmOp::getBInIndex());
  op->disconnectInTensor(in_b);
  auto B = in_b->id;
  if (gemm_op->getTransB()) {
    auto tB = ir.createIntermediateTensorId(B);
    transposeTensor(B, tB, op);
    B = tB;
  }

  // Create and connect up the matmul op.
  auto matmul = makeReplacementOpInIr(Onnx::AiOnnx::OpSet9::MatMul, op);
  matmul->connectInTensor(MatMulOp::getLhsInIndex(), A);
  matmul->connectInTensor(MatMulOp::getRhsInIndex(), B);
  matmul->createAndConnectOutTensor(MatMulOp::getOutIndex(),
                                    ir.createIntermediateTensorId(in_a->id));
  matmul->setup();

  if (op->hasInput(GemmOp::getCInIndex())) {
    // Scale the matmul by alpha.
    auto matmul_scale_out = ir.createIntermediateTensorId(in_a->id);
    scaleTensor(matmul->outTensor(MatMulOp::getOutIndex())->id,
                matmul_scale_out,
                gemm_op->getAlpha(),
                op);

    // Scale in_c by beta
    auto in_c           = op->inTensor(GemmOp::getCInIndex());
    auto in_c_scale_out = ir.createIntermediateTensorId(in_a->id);
    op->disconnectInTensor(in_c);
    auto beta = gemm_op->getBeta();
    scaleTensor(in_c->id, in_c_scale_out, beta, op);

    // Add the matmul output by the scaled in_c
    auto add = makeReplacementOpInIr(Onnx::AiOnnx::OpSet9::Add, op);
    add->connectInTensor(AddOp::getArg0InIndex(), matmul_scale_out);
    add->connectInTensor(AddOp::getArg1InIndex(), in_c_scale_out);
    add->connectOutTensor(AddOp::getOutIndex(), output->id);
  } else {
    // Scale the matmul by alpha.
    scaleTensor(matmul->outTensor(MatMulOp::getOutIndex())->id,
                output->id,
                gemm_op->getAlpha(),
                op);
  }

  if (op->input->n() > 0 || op->output->n() > 0) {
    throw internal_error(
        "All inputs and outputs of GemmOp should have been disconnected.");
  }
  // Remove the GemmOp
  op->getGraph().eraseOp(op->id);

  return true;
}

void GemmDecompositionPattern::scaleTensor(const TensorId &input,
                                           const TensorId &output,
                                           float scale_factor,
                                           Op *op) const {
  auto scale = dynamic_cast<ScaleOp *>(
      makeReplacementOpInIr(Onnx::AiGraphcore::OpSet1::Scale, op));
  scale->setScaleFactor(scale_factor);

  scale->connectInTensor(ScaleOp::getInIndex(), input);
  if (op->getGraph().getTensors().contains(output)) {
    scale->connectOutTensor(ScaleOp::getOutIndex(), output);
  } else {
    scale->createAndConnectOutTensor(ScaleOp::getOutIndex(), output);
  }

  scale->setup();
}

void GemmDecompositionPattern::transposeTensor(const TensorId &input,
                                               const TensorId &output,
                                               Op *op) const {
  std::vector<int64_t> perm{1, 0};

  auto transpose = dynamic_cast<TransposeOp *>(
      makeReplacementOpInIr(Onnx::AiOnnx::OpSet9::Transpose, op));
  transpose->setPerm(perm);

  transpose->connectInTensor(TransposeOp::getInIndex(), input);
  transpose->createAndConnectOutTensor(TransposeOp::getOutIndex(), output);
  transpose->setup();
}

namespace {
static PatternCreator<GemmDecompositionPattern>
    GemmDecompositionPattern(PreAliasPatternType::GemmDecomposition,
                             "GemmDecomposition");
}

} // namespace popart
