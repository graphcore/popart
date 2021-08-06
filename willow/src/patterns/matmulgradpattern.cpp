// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <memory>
#include <numeric>
#include <poprithms/ndarray/shape.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/reducesum.hpp>
#include <popart/op/reshape.hpp>
#include <popart/op/transpose.hpp>
#include <popart/patterns/matmulgradpattern.hpp>
#include <popart/patterns/patterns.hpp>
#include <popart/tensor.hpp>
#include <popart/tensors.hpp>
#include <popart/topocons.hpp>
#include <popart/util.hpp>

namespace popart {

namespace {

Tensor *configureMatMulOp(MatMulOp *op,
                          const TensorId lhsTensorId,
                          const TensorId rhsTensorId,
                          const TensorId outTensorId) {
  op->connectInTensor(MatMulOp::getLhsInIndex(), lhsTensorId);
  op->connectInTensor(MatMulOp::getRhsInIndex(), rhsTensorId);
  op->createAndConnectOutTensor(MatMulOp::getOutIndex(), outTensorId);
  op->setup();
  return op->outTensor(MatMulOp::getOutIndex());
}

std::vector<int64_t> getTransposeDimensions(popart::Tensor *t) {
  // Transpose the final two dimensions
  auto rank = t->info.rank();

  if (rank < 2) {
    throw error("Rank of input {} it too small for "
                "MatMulGradPattern::getTransposeDimensions",
                rank);
  }

  std::vector<int64_t> dims(rank);
  std::iota(dims.begin(), dims.end(), 0);
  std::swap(dims[rank - 2], dims[rank - 1]);
  return dims;
}

popart::Tensor *configureReshapeOp(ReshapeOp *op,
                                   const Shape &outShape,
                                   const TensorId inputTensorId,
                                   const TensorId outputTensorId) {
  op->setOutShape(outShape);
  op->connectInTensor(ReshapeOp::getInIndex(), inputTensorId);
  op->connectOutTensor(ReshapeOp::getOutIndex(), outputTensorId);
  op->setup();
  return op->outTensor(ReshapeOp::getOutIndex());
}

popart::Tensor *configureReshapeOp(ReshapeOp *op,
                                   const Shape &outShape,
                                   const TensorId inputTensorId) {
  op->setOutShape(outShape);
  op->connectInTensor(ReshapeOp::getInIndex(), inputTensorId);
  op->createAndConnectOutTensor(
      ReshapeOp::getOutIndex(),
      op->getIr().createIntermediateTensorId(inputTensorId));
  op->setup();
  return op->outTensor(ReshapeOp::getOutIndex());
}

popart::Tensor *configureTranposeOp(TransposeOp *op,
                                    const TensorId inputTensorId,
                                    const Shape &perm) {
  op->setPerm(perm);
  op->connectInTensor(TransposeOp::getInIndex(), inputTensorId);
  op->createAndConnectOutTensor(
      TransposeOp::getOutIndex(),
      op->getIr().createIntermediateTensorId(inputTensorId));
  op->setup();
  return op->outTensor(TransposeOp::getOutIndex());
}

popart::Tensor *configureReduceSumOp(ReduceSumOp *op,
                                     const TensorId inputTensorId,
                                     const Shape &axes,
                                     bool keepDims) {
  op->setAxes(axes);
  op->setKeepDims(keepDims);
  op->connectInTensor(ReduceSumOp::getInIndex(), inputTensorId);
  op->createAndConnectOutTensor(
      ReduceSumOp::getOutIndex(),
      op->getIr().createIntermediateTensorId(inputTensorId));
  op->setup();
  return op->outTensor(ReduceSumOp::getOutIndex());
}

popart::Tensor *configureSqueezeOp(ReshapeOp *op,
                                   const TensorId inputTensorId,
                                   const Shape &newShape) {
  op->setOutShape(newShape);
  op->connectInTensor(ReshapeOp::getInIndex(), inputTensorId);
  op->createAndConnectOutTensor(
      ReshapeOp::getOutIndex(),
      op->getIr().createIntermediateTensorId(inputTensorId));
  op->setup();
  return op->outTensor(ReshapeOp::getOutIndex());
}

Shape getLhsShape(Op *op) {
  MatMulBaseOp *matmulOp = dynamic_cast<MatMulBaseOp *>(op);
  return matmulOp->getExpandedLhsShape();
}
Shape getRhsShape(Op *op) {
  MatMulBaseOp *matmulOp = dynamic_cast<MatMulBaseOp *>(op);
  return matmulOp->getExpandedRhsShape();
}

} // namespace

bool MatMulPattern::matches(Op *op) const {
  if (op->opid == Onnx::Operators::MatMul_1 ||
      op->opid == Onnx::Operators::MatMul_9) {
    // If the inputs are less than 3d
    auto lhs = op->inTensor(MatMulOp::getLhsInIndex());
    auto rhs = op->inTensor(MatMulOp::getRhsInIndex());

    // Match if either inputs is not a minimum 3d tensor
    if (lhs->info.rank() >= 3 && rhs->info.rank() >= 3) {
      return false;
    } else {
      return true;
    }
  } else {
    return false;
  }
}

bool MatMulPattern::apply(Op *op) const {

  MatMulOp *matmulOp = dynamic_cast<MatMulOp *>(op);

  // The inputs/output tensors of the original matmul
  auto lhs = matmulOp->lhsIn();
  auto rhs = matmulOp->rhsIn();
  auto out = matmulOp->out();

  auto lhsReshapeOp = dynamic_cast<ReshapeOp *>(
      makeReplacementOpInIr(Onnx::Operators::Reshape_5, op, "LhsReshape"));
  auto rhsReshapeOp = dynamic_cast<ReshapeOp *>(
      makeReplacementOpInIr(Onnx::Operators::Reshape_5, op, "RhsReshape"));
  auto outReshapeOp = dynamic_cast<ReshapeOp *>(
      makeReplacementOpInIr(Onnx::Operators::Reshape_5, op, "OutReshape"));

  // Guaranteed to be a minimum of rank 3
  auto lhsShape = matmulOp->getExpandedLhsShape();
  auto rhsShape = matmulOp->getExpandedRhsShape();
  if (lhsShape.size() == 3 && rhsShape.size() == 3 && rhsShape[0] == 1) {
      // Modify the specific case where the RHS is broadcast on the group dimension.
      // This avoids a ReduceSumOp being inserted into the during autodiff.
      lhsShape[1] *= lhsShape[0];
      lhsShape[0] = 1;
  }

  logging::pattern::debug(
      "Applying MatMulOp pattern to reshape input from {} x {} to {} x {}",
      lhs->info.shape(),
      rhs->info.shape(),
      lhsShape,
      rhsShape);

  // expand the lhs input by reshaping it
  configureReshapeOp(lhsReshapeOp, lhsShape, lhs->id);

  // expand the rhs input by reshaping it
  configureReshapeOp(rhsReshapeOp, rhsShape, rhs->id);

  // disconnect the mat mul from it's original inputs & output
  matmulOp->disconnectAllInputs();
  matmulOp->disconnectAllOutputs();

  // Setup the new matmul for 3d inputs
  configureMatMulOp(matmulOp,
                    lhsReshapeOp->outTensor(ReshapeOp::getOutIndex())->id,
                    rhsReshapeOp->outTensor(ReshapeOp::getOutIndex())->id,
                    matmulOp->getIr().createIntermediateTensorId(out->id));

  // Reshape the output back the the user defined shape
  configureReshapeOp(outReshapeOp,
                     out->info.shape(),
                     matmulOp->outTensor(MatMulOp::getOutIndex())->id,
                     out->id);

  // Tie operations together.
  op->getGraph().topoCons->insert(lhsReshapeOp, matmulOp, true);
  op->getGraph().topoCons->insert(rhsReshapeOp, matmulOp, true);
  op->getGraph().topoCons->insert(matmulOp, outReshapeOp, true);

  return true;
}

bool MatMulGradPattern::apply(Op *op) const {

  auto in       = getIn(op);
  auto grad_in  = getGradIn(op);
  auto grad_out = getGradOut(op);

  // Get the phase of the matmul grad op
  auto phase = dynamic_cast<MatMulBaseOp *>(op)->getPhase();

  auto lhsShape = getLhsShape(op);
  auto rhsShape = getRhsShape(op);

  logging::pattern::info("Applying {} pattern", getPatternName());

  auto reshapeOpInExpand = dynamic_cast<ReshapeOp *>(
      makeReplacementOpInIr(Onnx::Operators::Reshape_5, op, "ReshapeIn"));
  auto reshapeOpGradInExpand = dynamic_cast<ReshapeOp *>(
      makeReplacementOpInIr(Onnx::Operators::Reshape_5, op, "ReshapeGradIn"));
  auto transposeOp = dynamic_cast<TransposeOp *>(
      makeReplacementOpInIr(Onnx::Operators::Transpose_1, op, "TransposeIn"));
  auto matmulOp = dynamic_cast<MatMulOp *>(
      makeReplacementOpInIr(Onnx::Operators::MatMul_9, op));

  // The Reshape and Transpose operations are on the path from the input of the
  // Matmul in the forward pass. They can therefore be scheduled before the
  // loss operation.
  // Therefore these Ops should not inherit 'toLoss' and 'fromLoss' properties
  // from the MatmulGradOp. Let's unset these properties, and leave it to
  // Ir::updateVertices to set these correctly.
  // Note that an exception is when doing explicit recomputation, which relies
  // on grad ops, when being decomposed as they are here, retaining these
  // properties. This is so it can correctly connect a computed tensor in the
  // IR to the operations in the backwards pass.
  if (op->getIr().getSessionOptions().explicitRecomputation == false) {
    reshapeOpInExpand->fromLoss = PathFromLoss::Undefined;
    reshapeOpInExpand->toLoss   = PathToLoss::Undefined;
    transposeOp->fromLoss       = PathFromLoss::Undefined;
    transposeOp->toLoss         = PathToLoss::Undefined;
  }

  // Copy over the matmul settings from original `matmulBaseOp`
  auto matmulBaseOp = dynamic_cast<MatMulBaseOp *>(op);
  matmulOp->setAvailableMemoryProportion(
      matmulBaseOp->getAvailableMemoryProportion());
  matmulOp->getSerialiseSettings() = (matmulBaseOp->getSerialiseSettings());
  matmulOp->setUseFullyConnectedPass(matmulBaseOp->useFullyConnectedPass());
  matmulOp->setPartialsType(matmulBaseOp->getPartialsType());

  auto squeezeOp = dynamic_cast<ReshapeOp *>(
      makeReplacementOpInIr(Onnx::Operators::Reshape_5, op, "Squeeze"));
  auto reduceSumOp = dynamic_cast<ReduceSumOp *>(
      makeReplacementOpInIr(Onnx::Operators::ReduceSum_1, op, "ReduceOut"));
  auto reshapeOp = dynamic_cast<ReshapeOp *>(
      makeReplacementOpInIr(Onnx::Operators::Reshape_5, op, "ReduceOut"));

  // Remove the MatMulXXXGradOp
  op->disconnectAllInputs();
  op->disconnectAllOutputs();
  // Make sure that any constrain on the matmulXXXgradop is moved to the matmul
  // op
  op->getGraph().topoCons->transfer(op, matmulOp);
  op->getGraph().eraseOp(op->id);

  // We will add the reshapes, if they have no effect they will be eliminated
  // from the ir Expand the grad_in tensor to a minimum 3d tensor
  {
    Shape grad_inShape;

    if (getGradInIndex() == MatMulOp::getRhsInIndex()) {
      grad_inShape = rhsShape;
    } else {
      grad_inShape = lhsShape;
    }

    // expand the input by reshaping it
    grad_in =
        configureReshapeOp(reshapeOpGradInExpand, grad_inShape, grad_in->id);
  }

  // Expand the in tensor to a minimum 3d tensor
  {
    Shape inShape;

    if (getInIndex() == MatMulOp::getRhsInIndex()) {
      inShape = rhsShape;
    } else {
      inShape = lhsShape;
    }

    // expand the input by reshaping it
    in = configureReshapeOp(reshapeOpInExpand, inShape, in->id);

    // Add constraint that we will not reshape the forward in tensor until the
    // grad_in tensor has been reshaped i.e. prevent the forward in reshape from
    // running in the forward pass.
    reshapeOpInExpand->getGraph().topoCons->insert(reshapeOpGradInExpand,
                                                   reshapeOpInExpand);
  }

  // Configure the tranpose the in tensor
  in = configureTranposeOp(transposeOp, in->id, getTransposeDimensions(in));

  // Configure the mat mul op
  matmulOp->setCanCreateInputs(false);
  matmulOp->connectInTensor(getGradInIndex(), grad_in->id);
  matmulOp->connectInTensor(getInIndex(), in->id);
  matmulOp->createAndConnectOutTensor(
      MatMulOp::getOutIndex(),
      matmulOp->getIr().createIntermediateTensorId(grad_out->id));
  matmulOp->setPhase(phase);
  matmulOp->setup();
  auto out = matmulOp->outTensor(MatMulOp::getOutIndex());

  if (out->info.shape() == grad_out->info.shape()) {
    // The output of matmul is correct, remove the intermediate tensor and
    // instead use the grad_out
    matmulOp->disconnectAllOutputs();
    matmulOp->connectOutTensor(MatMulOp::getOutIndex(), grad_out->id);

  } else {

    logging::pattern::debug("{} need to reduce {} to {}",
                            getPatternName(),
                            out->info.shape(),
                            grad_out->info.shape());
    // The output of the matmul needs to be reduced to match the expected output
    auto matmulOutputShape = out->info.shape();

    // First remove any leading '1' dimension with a squeeze
    std::vector<int64_t> squeezeDims;
    for (auto i = 0; i < out->info.shape().size(); ++i) {
      if (out->info.shape()[i] == 1) {
        squeezeDims.push_back(i);
      } else {
        break;
      }
    }

    if (squeezeDims.size() > 0) {
      matmulOutputShape = squeeze(out->info.shape(), squeezeDims);
    }

    if (matmulOutputShape != grad_out->info.shape()) {

      if (grad_out->info.shape().size() == 1 && matmulOutputShape.size() > 0) {

        for (int i = 0; i < out->info.shape().size(); i++) {
          if (out->info.shape()[out->info.shape().size() - 1 - i] == 1) {
            squeezeDims.push_back(out->info.shape().size() - 1 - i);
          } else {
            break;
          }
        }
      }
    }

    if (squeezeDims.size() > 0) {
      logging::pattern::debug("{} squeezing 1's {} from {}",
                              getPatternName(),
                              squeezeDims,
                              out->info.shape());

      auto newShape = poprithms::ndarray::Shape(out->info.shape())
                          .squeeze({squeezeDims.cbegin(), squeezeDims.cend()});
      out = configureSqueezeOp(squeezeOp, out->id, newShape.get());
    }

    if (out->info.shape() == grad_out->info.shape()) {
      // The output of transpose/matmul/squeeze is correct, remove the
      // intermediate tensor and instead use the grad_out
      squeezeOp->disconnectAllOutputs();
      squeezeOp->connectOutTensor(ReshapeOp::getOutIndex(), grad_out->id);
    } else {

      // The shapes are still not the same then use the reduce
      popart::Shape targetShape = grad_out->info.shape();
      popart::Shape outShape    = out->info.shape();

      // prepend 1's to the outShape
      while (outShape.size() < targetShape.size()) {
        outShape.insert(outShape.begin(), 1);
      }

      while (outShape.size() > targetShape.size()) {
        targetShape.insert(targetShape.begin(), 1);
      }

      std::vector<std::int64_t> reduceDims;
      for (int i = 0; i < outShape.size(); ++i) {
        if (outShape[i] != targetShape[i]) {
          reduceDims.push_back(i);
        }
      }

      // Reduce & reshape the output
      out = configureReduceSumOp(reduceSumOp, out->id, reduceDims, false);
      configureReshapeOp(
          reshapeOp, grad_out->info.shape(), out->id, grad_out->id);
    }
  }

  // Tie operations together
  matmulOp->getGraph().topoCons->insert(reshapeOpInExpand, matmulOp, true);
  matmulOp->getGraph().topoCons->insert(reshapeOpGradInExpand, matmulOp, true);
  matmulOp->getGraph().topoCons->insert(transposeOp, matmulOp, true);
  matmulOp->getGraph().topoCons->insert(matmulOp, squeezeOp, true);
  matmulOp->getGraph().topoCons->insert(matmulOp, reduceSumOp, true);
  matmulOp->getGraph().topoCons->insert(matmulOp, reshapeOp, true);

  // Remove any ops not used
  auto removedIfNotUsed = [](Op *opToRemove) {
    if (opToRemove->inTensorCount() == 0) {
      opToRemove->getGraph().eraseOp(opToRemove->id);
    }
  };

  removedIfNotUsed(reshapeOpInExpand);
  removedIfNotUsed(reshapeOpGradInExpand);
  removedIfNotUsed(squeezeOp);
  removedIfNotUsed(reduceSumOp);
  removedIfNotUsed(reshapeOp);

  return true;
}

bool MatMulLhsGradPattern::matches(Op *op) const {
  return (op->opid == Onnx::GradOperators::MatMulLhsGrad);
}

bool MatMulRhsGradPattern::matches(Op *op) const {
  return (op->opid == Onnx::GradOperators::MatMulRhsGrad);
}

namespace {
static PatternCreator<MatMulPattern>
    matMulPattern(PreAliasPatternType::MatMulOp, "MatMulOp", true);
static PatternCreator<MatMulLhsGradPattern>
    matMulLhsGradPattern(PreAliasPatternType::MatMulLHSGradOp,
                         "MatMulLhsGradOp",
                         true);
static PatternCreator<MatMulRhsGradPattern>
    matMulRhsGradPattern(PreAliasPatternType::MatMulRHSGradOp,
                         "MatMulRhsGradOp",
                         true);
} // namespace

} // namespace popart
