#include <poponnx/ir.hpp>
#include <poponnx/makeunique.hpp>
#include <poponnx/op/add.hpp>
#include <poponnx/op/gemm.hpp>
#include <poponnx/op/matmul.hpp>
#include <poponnx/op/scale.hpp>
#include <poponnx/op/transpose.hpp>
#include <poponnx/patterns/gemmdecompositionpattern.hpp>
#include <poponnx/tensor.hpp>
#include <poponnx/tensorinfo.hpp>

namespace poponnx {

bool GemmDecompositionPattern::matches(Op *op) const {
  return op->isConvertibleTo<GemmOp>();
}

std::vector<const Tensor *> GemmDecompositionPattern::touches(Op *) const {
  return {};
}

// output = add(scale(matmul(in_a, in_b), alpha), scale(in_c, beta))
bool GemmDecompositionPattern::apply(Op *op) const {
  auto in_a   = op->inTensor(GemmOp::getAInIndex());
  auto in_b   = op->inTensor(GemmOp::getBInIndex());
  auto in_c   = op->inTensor(GemmOp::getCInIndex());
  auto output = op->outTensor(GemmOp::getOutIndex());

  // we assume this dynamic_cast call has been confirmed
  // to be valid via a previous call to GemmDecompositionPattern::matches
  auto gemm_op = dynamic_cast<GemmOp *>(op);
  auto alpha   = gemm_op->getAlpha();
  auto beta    = gemm_op->getBeta();
  auto transA  = gemm_op->getTransA();
  auto transB  = gemm_op->getTransB();

  // create the new ops
  auto matmul = makeReplacementOpInIr(Onnx::AiOnnx::OpSet9::MatMul, op);
  auto add    = makeReplacementOpInIr(Onnx::AiOnnx::OpSet9::Add, op);

  auto A = in_a->id;
  if (transA) {
    auto tA = createIntermediateTensorId(A);
    transposeTensor(A, tA, op);
    A = tA;
  }

  auto B = in_b->id;
  if (transB) {
    auto tB = createIntermediateTensorId(B);
    transposeTensor(B, tB, op);
    B = tB;
  }

  // Connect up the new ops
  matmul->connectInTensor(MatMulOp::getLhsInIndex(), A);
  matmul->connectInTensor(MatMulOp::getRhsInIndex(), B);
  matmul->createAndConnectOutTensor(MatMulOp::getOutIndex(),
                                    createIntermediateTensorId(in_a->id));
  matmul->setup();

  auto scale1_out = createIntermediateTensorId(in_a->id);
  scaleTensor(
      matmul->outTensor(MatMulOp::getOutIndex())->id, scale1_out, alpha, op);

  auto scale2_out = createIntermediateTensorId(in_a->id);
  scaleTensor(in_c->id, scale2_out, beta, op);

  add->connectInTensor(AddOp::getArg0InIndex(), scale1_out);
  add->connectInTensor(AddOp::getArg1InIndex(), scale2_out);
  add->connectOutTensor(AddOp::getOutIndex(), output->id);

  // Remove the GemmOp
  op->disconnectAllInputs();
  op->disconnectAllOutputs();
  op->pir->eraseOp(op->id);

  return true;
}

void GemmDecompositionPattern::scaleTensor(const TensorId &input,
                                           const TensorId &output,
                                           float scale_factor,
                                           Op *op) const {
  auto scale = dynamic_cast<ScaleOp *>(
      makeReplacementOpInIr(Onnx::AiOnnx::OpSet9::Scale, op));
  scale->setScaleFactor(scale_factor);

  scale->connectInTensor(ScaleOp::getInIndex(), input);
  scale->createAndConnectOutTensor(ScaleOp::getOutIndex(), output);
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
    GemmDecompositionPattern(PatternType::GEMMDECOMPOSITION,
                             "GemmDecomposition");
}

} // namespace poponnx
