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

static void transposeTensor(const TensorId &input,
                            const TensorId &output,
                            Ir *ir,
                            Attributes attr);
static void scaleTensor(const TensorId &input,
                        const TensorId &output,
                        float scale_factor,
                        Ir *ir,
                        Attributes attr);

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

  auto ir   = op->pir;
  auto attr = op->nAtts.filter(sVirtualGraphAttribute);

  // create the new ops
  auto matmul_op =
      make_unique<MatMulOp>(Onnx::Operators::MatMul, ir, std::string{}, attr);
  auto add_op =
      make_unique<AddOp>(Onnx::Operators::Add, ir, std::string{}, attr);

  // move ops into ir
  auto matmul = matmul_op.get();
  auto add    = add_op.get();
  ir->moveIntoIr(std::move(matmul_op));
  ir->moveIntoIr(std::move(add_op));

  // Remove the GemmOp
  op->disconnectAllInputs();
  op->disconnectAllOutputs();
  ir->eraseOp(op->id);

  auto A = in_a->id;
  if (transA) {
    auto tA = createIntermediateTensorId(A);
    transposeTensor(A, tA, ir, attr);
    A = tA;
  }

  auto B = in_b->id;
  if (transB) {
    auto tB = createIntermediateTensorId(B);
    transposeTensor(B, tB, ir, attr);
    B = tB;
  }

  // Connect up the new ops
  matmul->connectInTensor(MatMulOp::getLhsInIndex(), A);
  matmul->connectInTensor(MatMulOp::getRhsInIndex(), B);
  matmul->createAndConnectOutTensor(MatMulOp::getOutIndex(),
                                    createIntermediateTensorId(in_a->id));
  matmul->setup();

  auto scale1_out = createIntermediateTensorId(in_a->id);
  scaleTensor(matmul->outTensor(MatMulOp::getOutIndex())->id,
              scale1_out,
              alpha,
              ir,
              attr);

  auto scale2_out = createIntermediateTensorId(in_a->id);
  scaleTensor(in_c->id, scale2_out, beta, ir, attr);

  add->connectInTensor(AddOp::getArg0InIndex(), scale1_out);
  add->connectInTensor(AddOp::getArg1InIndex(), scale2_out);
  add->connectOutTensor(AddOp::getOutIndex(), output->id);

  return true;
}

static void scaleTensor(const TensorId &input,
                        const TensorId &output,
                        float scale_factor,
                        Ir *ir,
                        Attributes attr) {
  auto scale_op =
      make_unique<ScaleOp>(Onnx::Operators::Scale, ir, std::string{}, attr);
  scale_op->setScaleFactor(scale_factor);

  auto scale = scale_op.get();
  ir->moveIntoIr(std::move(scale_op));

  scale->connectInTensor(ScaleOp::getInIndex(), input);
  scale->createAndConnectOutTensor(ScaleOp::getOutIndex(), output);
  scale->setup();
}

static void transposeTensor(const TensorId &input,
                            const TensorId &output,
                            Ir *ir,
                            Attributes attr) {
  std::vector<int64_t> perm{1, 0};

  auto transpose_op = make_unique<TransposeOp>(
      Onnx::Operators::Transpose, ir, std::string{}, attr);
  transpose_op->setPerm(perm);

  auto transpose = transpose_op.get();
  ir->moveIntoIr(std::move(transpose_op));

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
