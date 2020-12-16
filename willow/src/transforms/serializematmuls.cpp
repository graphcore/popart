// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <memory>
#include <popart/error.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/logging.hpp>
#include <popart/names.hpp>
#include <popart/op.hpp>
#include <popart/op/concat.hpp>
#include <popart/op/identity.hpp>
#include <popart/op/ipucopy.hpp>
#include <popart/op/matmul.hpp>
#include <popart/op/reshape.hpp>
#include <popart/op/slice.hpp>
#include <popart/op/transpose.hpp>
#include <popart/op/varupdate.hpp>

#include <popart/optimizer.hpp>
#include <popart/tensor.hpp>
#include <popart/tensors.hpp>
#include <popart/topocons.hpp>
#include <popart/util.hpp>

#include <popart/transforms/serializematmuls.hpp>
#include <popart/transforms/transformbuilder.hpp>

#include <boost/optional/optional_io.hpp>
#include <boost/range/algorithm_ext.hpp>
#include <popart/vendored/any.hpp>

// X = [GROUP_DIM, INPUT_CHANNELS, REDUCING_DIM]
// W = [GROUP_DIM, REDUCING_DIM, OUTPUT_CHANNELS]

// If 'keep_precision' is specified then any MatMul that is split along it's
// reducing dimension will have an output type of FLOAT and a cast will be added
// after the addInplaces.
//
// For D = INPUT_CHANNELS
//
//      X   W            d_Y     W.T         X.T    d_Y
//      |   |             |      |            |      |
//      MatMulOp      MatMulLhsGradOp       MatMulRhsGradOp
//        |                  |                  |
//        Y                 d_X                d_W
//
//
//                       Transform to:
//
//  For n in [0,N):
//  |
//  |  X[n:n+1,:] W     d_Y[:,n:n+1] W.T           X.T[:,n:n+1] d_Y[n:n+1,:]
//  |      |      |       |         |                |         |
//  |      MatMulOp      MatMulLhsGradOp           MatMulRhsGradOp
//  |        |                  |                      |
//  |       Y_n               d_X_n                   d_W_n
//  |
//  | _d_W = addInplace(d_W_n-1, d_W_n)
//
//  Y    = concat(Y_0...Y_n)
//  d_x  = concat(d_X_0...d_X_n)
//  d_W  = cast(_d_W) if type(d_W) != type(_d_W) else _d_W
//

//  For D = OUTPUT_CHANNEL
//
//      X   W            d_Y     W.T         X.T    d_Y
//      |   |             |      |            |      |
//      MatMulOp      MatMulLhsGradOp       MatMulRhsGradOp
//        |                  |                  |
//        Y                 d_X                d_W    W
//                                              |     |
//                                             VarUpdate
//
//                       Transform to:
//
//  For n in [0,N):
//  |
//  |     X  W[:,n:n+1]  d_Y[:,n:n+1]  W.T[n:n+1,:] X.T    d_Y[:,n:n+1]
//  |      |      |       |         |                |      |
//  |      MatMulOp      MatMulLhsGradOp           MatMulRhsGradOp
//  |        |                  |                      |
//  |       Y_n               d_X_n                   d_W_n  W[:,n:n+1]
//  |                                                  |     |
//  |                                                 VarUpdate
//  | _d_x = addInplace(d_x_n-1, d_x_n)
//
//  Y    = concat(Y_0...Y_n) - concat on final dim
//  d_X  = cast(_d_X) if type(d_X) != type(_d_X) else _d_X
//

//                    For D = REDUCING_DIM
//      X   W            d_Y     W.T         X.T    d_Y
//      |   |             |      |            |      |
//      MatMulOp      MatMulLhsGradOp       MatMulRhsGradOp
//        |                  |                  |
//        Y                 d_X                d_W    W
//                                              |     |
//                                             VarUpdate
//
//                       Transform to:
//
//  For n in [0,N):
//  |
//  | X[:,n:n+1] W[n:n+1,:] d_Y  W.T[:,n:n+1]       X.T[n:n+1,:] d_Y
//  |      |      |          |         |                |         |
//  |      MatMulOp         MatMulLhsGradOp           MatMulRhsGradOp
//  |        |                     |                      |
//  |       Y_n                  d_X_n                   d_W_n  W[n:n+1,:]
//  |                                                     |     |
//  |                                                    VarUpdate
//  | _Y = addInplace(Y_n-1, Y_n)
//
//  Y    = cast(_Y) if type(Y) != type(_Y) else _Y
//  d_X  = concat(d_X_0...d_X_n)

namespace {

// Find all matmuls in the graph which have serialization enabled
std::vector<popart::MatMulBaseOp *>
findSerializedMatMuls(popart::Graph &graph) {

  std::vector<popart::MatMulBaseOp *> matmuls;

  for (auto &entry : graph.getOps()) {
    auto *op = entry.second.get();

    auto *matmul = dynamic_cast<popart::MatMulBaseOp *>(op);
    if (matmul != nullptr) {
      if (matmul->getSerialiseSettings().mode !=
          popart::MatMulBaseOp::SerialiseSettings::Mode::None) {
        matmuls.push_back(matmul);
      }
    }
  }

  std::sort(matmuls.begin(),
            matmuls.end(),
            [](const popart::Op *op0, const popart::Op *op1) {
              return op0->id < op1->id;
            });

  return matmuls;
}
} // namespace

namespace popart {

std::size_t SerializeMatMuls::id() {
  return typeid(SerializeMatMuls).hash_code();
}

static bool keepPrecision(MatMulBaseOp *matmul) {
  return matmul->getSerialiseSettings().keep_precision &&
         matmul->getPartialsType() == MatMulPartialsType::FLOAT;
}

static void serializeMatMul(TransformBuilder &builder,
                            Tensor *lhs,
                            int sliceLhsDim, // -1 indicates no slice
                            Tensor *rhs,
                            int sliceRhsDim, // -1 indicates no slice
                            Tensor *output,
                            MatMulBaseOp *matmul,
                            std::vector<TensorId> &outputTensors,
                            OptionalVGraphId virtualGraphId,
                            OptionalPipelineStage pipelineStage,
                            OptionalExecutionPhase executionPhase,
                            std::string name) {

  for (int i = 0; i < matmul->getSerialiseSettings().factor; ++i) {

    TensorId lhsMatMulInput = lhs->id;
    TensorId rhsMatMulInput = rhs->id;

    if (sliceLhsDim != -1) {

      int sliceDim = sliceLhsDim;
      auto size =
          lhs->info.dim(sliceDim) / matmul->getSerialiseSettings().factor;

      Shape starts = {(i)*size};
      Shape ends   = {(i + 1) * size};
      Shape axes   = {sliceDim};

      lhsMatMulInput = builder.slice(lhs->id,
                                     starts,
                                     ends,
                                     axes,
                                     virtualGraphId,
                                     pipelineStage,
                                     executionPhase,
                                     name + "_SliceLhs",
                                     builder.getNextId(name + "_SliceLhs"));
    }

    if (sliceRhsDim != -1) {

      int sliceDim = sliceRhsDim;
      auto size =
          rhs->info.dim(sliceDim) / matmul->getSerialiseSettings().factor;

      Shape starts = {(i)*size};
      Shape ends   = {(i + 1) * size};
      Shape axes   = {sliceDim};

      rhsMatMulInput = builder.slice(rhs->id,
                                     starts,
                                     ends,
                                     axes,
                                     virtualGraphId,
                                     pipelineStage,
                                     executionPhase,
                                     name + "_SliceRhs",
                                     builder.getNextId(name + "_SliceRhs"));
    }

    std::map<std::string, popart::any> attrs = {};
    if (sliceLhsDim == 2 && sliceRhsDim == 1 && keepPrecision(matmul) &&
        output->info.dataType() != DataType::FLOAT) {
      attrs.insert({sOutputTypeAttribute, std::string("FLOAT")});
    }
    std::string partialType =
        matmul->getPartialsType() == MatMulPartialsType::FLOAT ? "FLOAT"
                                                               : "HALF";
    attrs.insert({sPartialsTypeAttribute, partialType});

    auto m = builder.matmul(lhsMatMulInput,
                            rhsMatMulInput,
                            virtualGraphId,
                            pipelineStage,
                            executionPhase,
                            name + "_MatMul",
                            builder.getNextId(name + "_MatMul"),
                            attrs,
                            matmul->opid);

    auto *mOp = dynamic_cast<MatMulBaseOp *>(builder.getProducer(m));

    mOp->setPhase(matmul->getPhase());
    mOp->setAvailableMemoryProportion(matmul->getAvailableMemoryProportion());
    mOp->setUseFullyConnectedPass(matmul->useFullyConnectedPass());
    if (builder.hasProducer(lhsMatMulInput))
      builder.getGraph().topoCons->insert(
          builder.getProducer(lhsMatMulInput), mOp, true);
    if (builder.hasProducer(rhsMatMulInput))
      builder.getGraph().topoCons->insert(
          builder.getProducer(rhsMatMulInput), mOp, true);

    outputTensors.push_back(m);
    if (i > 0)
      builder.getGraph().topoCons->insert(
          builder.getProducer(outputTensors[i - 1]),
          builder.getProducer(outputTensors[i]));
  }
}

static void sumByAddInplace(TransformBuilder &builder,
                            const bool cast_needed,
                            Tensor *output,
                            std::vector<TensorId> &outputTensors,
                            OptionalVGraphId virtualGraphId,
                            OptionalPipelineStage pipelineStage,
                            OptionalExecutionPhase executionPhase,
                            std::string name) {
  auto out = outputTensors[0];
  for (size_t i = 1; i < outputTensors.size(); i++) {
    std::vector<TensorId> inputs = {out, outputTensors[i]};
    if (!cast_needed && i == outputTensors.size() - 1) {
      builder.addLhsInplace(inputs,
                            output->id,
                            virtualGraphId,
                            pipelineStage,
                            executionPhase,
                            name + "_AddInplace");
      if (builder.hasProducer(outputTensors[i]) &&
          builder.hasProducer(output->id))
        builder.getGraph().topoCons->insert(
            builder.getProducer(outputTensors[i]),
            builder.getProducer(output->id),
            true);
    } else {
      out = builder.addLhsInplace(inputs,
                                  virtualGraphId,
                                  pipelineStage,
                                  executionPhase,
                                  name + "_AddInplace",
                                  builder.getNextId(name + "_AddInPlace"));
      if (builder.hasProducer(outputTensors[i]) && builder.hasProducer(out))
        builder.getGraph().topoCons->insert(
            builder.getProducer(outputTensors[i]),
            builder.getProducer(out),
            true);
      // Add a constraint so the inplace add happens before the next matmul
      if (!cast_needed)
        builder.getGraph().topoCons->insert(
            builder.getProducer(out),
            builder.getProducer(outputTensors[i + 1]));
    }
  }
  if (cast_needed) {
    logging::transform::debug(
        "Casting Output {} to {}", out, output->info.data_type());
    builder.cast(out,
                 output->id,
                 output->info.dataType(),
                 virtualGraphId,
                 pipelineStage,
                 executionPhase,
                 name + "_Cast");
    if (builder.hasProducer(out) && builder.hasProducer(output->id))
      builder.getGraph().topoCons->insert(
          builder.getProducer(out), builder.getProducer(output->id), true);
  }
}

static void serializeVarUpdate(int sliceDim,
                               TransformBuilder &builder,
                               Tensor *matMulOutput,
                               MatMulBaseOp *matmul,
                               std::vector<TensorId> &outputTensors,
                               OptionalVGraphId virtualGraphId,
                               OptionalPipelineStage pipelineStage,
                               OptionalExecutionPhase executionPhase,
                               std::string name) {
  auto chaseme = matMulOutput;
  std::vector<Op *> path;
  bool validPath               = true;
  bool endOfPathFound          = false;
  const int slice_dim_from_end = 3 - sliceDim;

  while (endOfPathFound == false && validPath == true) {

    auto consumerOps = chaseme->consumers.getOps();

    if (consumerOps.size() > 1) {
      logging::transform::warn("Do not currently support serializing ops for "
                               "tensors which have more than 1 consumer");
      validPath = false;
    } else {

      // Only certain ops can be on the path between matmul and varupdate.
      if (consumerOps[0]->isConvertibleTo<VarUpdateOp>() ||
          consumerOps[0]->isConvertibleTo<IdentityOp>() ||
          consumerOps[0]->isConvertibleTo<TransposeBaseOp>() ||
          consumerOps[0]->opid == Onnx::Operators::Reshape_1 ||
          consumerOps[0]->opid == Onnx::Operators::Reshape_5 ||
          consumerOps[0]->opid == Onnx::GradOperators::ReshapeGrad ||
          consumerOps[0]->opid == Onnx::Operators::ReduceSum_1) {

        // Add to the path
        path.push_back(consumerOps[0]);

        if (consumerOps[0]->isConvertibleTo<VarUpdateOp>()) {
          endOfPathFound = true;
        } else {
          // May need check here
          chaseme = consumerOps[0]->output->tensorMap().at(0);
        }
      } else {
        logging::transform::warn(
            "Do not currently support {} when serializing matmuls",
            consumerOps[0]->opid);
        validPath = false;
      }
    }
  }

  if (validPath) {

    // If we have a valid path we can serialise the operation on that path so we
    // can serialize the varupdate.

    auto varUpdate = dynamic_cast<VarUpdateOp *>(path.back());

    // Get the weight tensor
    auto weightTensor =
        varUpdate->input->tensorMap().at(VarUpdateOp::getVarToUpdateInIndex());

    // Calculate the size of serialized tensor
    auto size =
        weightTensor->info.dim(weightTensor->info.rank() - slice_dim_from_end) /
        matmul->getSerialiseSettings().factor;

    for (int i = 0; i < matmul->getSerialiseSettings().factor; ++i) {

      Shape starts = {(i)*size};
      Shape ends   = {(i + 1) * size};
      Shape axes   = {weightTensor->info.rank() - slice_dim_from_end};

      TensorId output = outputTensors[i];
      bool notFinal   = i != (matmul->getSerialiseSettings().factor - 1);

      // serialize the operations along the path
      for (auto op : path) {
        if (op->opid == Onnx::Operators::Reshape_1 ||
            op->opid == Onnx::Operators::Reshape_5 ||
            op->opid == Onnx::GradOperators::ReshapeGrad) {

          // detemine the new reshape 'shape' after serilization
          auto outputshape = op->output->tensorMap()
                                 .at(ReshapeBaseOp::getOutIndex())
                                 ->info.shape();
          auto d = outputshape.size() - slice_dim_from_end;
          outputshape[d] =
              outputshape[d] / matmul->getSerialiseSettings().factor;

          logging::op::debug("Serializing reshape {}", outputshape);

          output = builder.reshape(output,
                                   outputshape,
                                   virtualGraphId,
                                   pipelineStage,
                                   executionPhase,
                                   name + "_Reshape",
                                   builder.getNextId(name + "_Reshape"));
        } else if (op->opid == Onnx::Operators::ReduceSum_1) {
          logging::op::debug("Serializing reduced sum");
          output = builder.reducesum(output,
                                     0,
                                     {0},
                                     virtualGraphId,
                                     pipelineStage,
                                     executionPhase,
                                     name + "_ReduceSum",
                                     builder.getNextId(name + "_ReduceSum"));
        } else if (op->isConvertibleTo<TransposeBaseOp>()) {

          logging::op::debug("Serializing transpose {}", output);
          auto transposeBase = dynamic_cast<TransposeBaseOp *>(op);
          auto transposePerm = transposeBase->getPerm();

          output = builder.transpose(output,
                                     transposePerm,
                                     virtualGraphId,
                                     pipelineStage,
                                     executionPhase,
                                     name + "_Transpose",
                                     builder.getNextId(name + "_Transpose"));
          // Output slice dim changes after the transpose
          axes[0] = transposePerm[axes[0]];
          size    = weightTensor->info.dim(axes[0]) /
                 matmul->getSerialiseSettings().factor;
          starts = {(i)*size};
          ends   = {(i + 1) * size};

        } else if (op->isConvertibleTo<IdentityOp>()) {
          // Don't do anything
        } else if (op->isConvertibleTo<VarUpdateOp>()) {
          // Don't do anything
        } else {
          throw error("Do not support {} when serializing", op->opid);
        }
      }

      auto slicedWeight =
          builder.sliceInPlace(weightTensor->id,
                               starts,
                               ends,
                               axes,
                               virtualGraphId,
                               pipelineStage,
                               executionPhase,
                               name + "_Slice",
                               builder.getNextId(name + "_Slice"));

      // Make sure the slice the weight after the output has been serialized
      builder.getGraph().topoCons->insert(builder.getProducer(output),
                                          builder.getProducer(slicedWeight));

      // Create the new var update op

      auto slicedVarUpdateOp =
          dynamic_cast<VarUpdateOp *>(varUpdate)->cloneWithNewName(
              slicedWeight);

      for (auto &x : varUpdate->optimizerInputs()) {
        slicedVarUpdateOp->connectInTensor(x.first, x.second);
      }
      slicedVarUpdateOp->connectInTensor(VarUpdateOp::getVarToUpdateInIndex(),
                                         slicedWeight);
      slicedVarUpdateOp->connectInTensor(
          VarUpdateWithUpdaterOp::getUpdaterInIndex(), output);
      slicedVarUpdateOp->createAndConnectOutTensor(
          VarUpdateOp::getUpdatedVarOutIndex(), "updated__" + slicedWeight);
      if (virtualGraphId)
        slicedVarUpdateOp->setVirtualGraphId(*virtualGraphId);
      if (pipelineStage)
        slicedVarUpdateOp->setPipelineStage(*pipelineStage);
      slicedVarUpdateOp->setup();

      auto slicedVarUpdatePtr = slicedVarUpdateOp.get();

      builder.getGraph().moveIntoGraph(std::move(slicedVarUpdateOp));

      builder.getGraph().topoCons->transfer(varUpdate, slicedVarUpdatePtr);

      if (notFinal) {
        builder.getGraph().topoCons->insert(
            slicedVarUpdatePtr, builder.getProducer(outputTensors[i + 1]));
      }
    }

    // Remove the ops on the original path
    for (Op *opToRemove : path) {
      opToRemove->disconnectAllInputs();
      opToRemove->disconnectAllOutputs();
      builder.getGraph().eraseOp(opToRemove->id);
    }
  } else {
    // Concatenate the outputs
    builder.concat(outputTensors,
                   sliceDim,
                   matMulOutput->id,
                   virtualGraphId,
                   pipelineStage,
                   executionPhase,
                   name + "_Concat");
  }
}

static void
serializeFwdMatMul_InputChannels(TransformBuilder &builder,
                                 Tensor *lhs,
                                 Tensor *rhs,
                                 Tensor *output,
                                 MatMulBaseOp *matmul,
                                 std::vector<TensorId> &outputTensors,
                                 OptionalVGraphId virtualGraphId,
                                 OptionalPipelineStage pipelineStage,
                                 OptionalExecutionPhase executionPhase,
                                 std::string name) {
  serializeMatMul(builder,
                  lhs,
                  1,
                  rhs,
                  -1,
                  output,
                  matmul,
                  outputTensors,
                  virtualGraphId,
                  pipelineStage,
                  executionPhase,
                  name);

  builder.concat(outputTensors,
                 1,
                 output->id,
                 virtualGraphId,
                 pipelineStage,
                 executionPhase,
                 name + "_Concat");

  builder.getGraph().topoCons->transfer(matmul, output->getProducer());
}

static void
serializeBwdLhsMatMul_InputChannels(TransformBuilder &builder,
                                    Tensor *lhs,
                                    Tensor *rhs,
                                    Tensor *output,
                                    MatMulBaseOp *matmul,
                                    std::vector<TensorId> &outputTensors,
                                    OptionalVGraphId virtualGraphId,
                                    OptionalPipelineStage pipelineStage,
                                    OptionalExecutionPhase executionPhase,
                                    std::string name) {

  serializeMatMul(builder,
                  lhs,
                  1,
                  rhs,
                  -1,
                  output,
                  matmul,
                  outputTensors,
                  virtualGraphId,
                  pipelineStage,
                  executionPhase,
                  name);

  builder.concat(outputTensors,
                 1,
                 output->id,
                 virtualGraphId,
                 pipelineStage,
                 executionPhase,
                 name + "_Concat");
}

static void
serializeBwdRhsMatMul_InputChannels(TransformBuilder &builder,
                                    Tensor *lhs,
                                    Tensor *rhs,
                                    Tensor *output,
                                    MatMulBaseOp *matmul,
                                    std::vector<TensorId> &outputTensors,
                                    OptionalVGraphId virtualGraphId,
                                    OptionalPipelineStage pipelineStage,
                                    OptionalExecutionPhase executionPhase,
                                    std::string name) {
  serializeMatMul(builder,
                  lhs,
                  2,
                  rhs,
                  1,
                  output,
                  matmul,
                  outputTensors,
                  virtualGraphId,
                  pipelineStage,
                  executionPhase,
                  name);

  sumByAddInplace(builder,
                  keepPrecision(matmul),
                  output,
                  outputTensors,
                  virtualGraphId,
                  pipelineStage,
                  executionPhase,
                  name);
}

static void
serializeFwdMatMul_ReducingDim(TransformBuilder &builder,
                               Tensor *lhs,
                               Tensor *rhs,
                               Tensor *output,
                               MatMulBaseOp *matmul,
                               std::vector<TensorId> &outputTensors,
                               OptionalVGraphId virtualGraphId,
                               OptionalPipelineStage pipelineStage,
                               OptionalExecutionPhase executionPhase,
                               std::string name) {

  serializeMatMul(builder,
                  lhs,
                  2,
                  rhs,
                  1,
                  output,
                  matmul,
                  outputTensors,
                  virtualGraphId,
                  pipelineStage,
                  executionPhase,
                  name);

  sumByAddInplace(builder,
                  keepPrecision(matmul),
                  output,
                  outputTensors,
                  virtualGraphId,
                  pipelineStage,
                  executionPhase,
                  name);

  builder.getGraph().topoCons->transfer(matmul, output->getProducer());
}

static void
serializeBwdLhsMatMul_ReducingDim(TransformBuilder &builder,
                                  Tensor *lhs,
                                  Tensor *rhs,
                                  Tensor *output,
                                  MatMulBaseOp *matmul,
                                  std::vector<TensorId> &outputTensors,
                                  OptionalVGraphId virtualGraphId,
                                  OptionalPipelineStage pipelineStage,
                                  OptionalExecutionPhase executionPhase,
                                  std::string name) {

  serializeMatMul(builder,
                  lhs,
                  -1,
                  rhs,
                  2,
                  output,
                  matmul,
                  outputTensors,
                  virtualGraphId,
                  pipelineStage,
                  executionPhase,
                  name);

  builder.concat(outputTensors,
                 2,
                 output->id,
                 virtualGraphId,
                 pipelineStage,
                 executionPhase,
                 name + "_Concat");
}

static void
serializeBwdRhsMatMul_ReducingDim(TransformBuilder &builder,
                                  Tensor *lhs,
                                  Tensor *rhs,
                                  Tensor *output,
                                  MatMulBaseOp *matmul,
                                  std::vector<TensorId> &outputTensors,
                                  OptionalVGraphId virtualGraphId,
                                  OptionalPipelineStage pipelineStage,
                                  OptionalExecutionPhase executionPhase,
                                  std::string name) {
  serializeMatMul(builder,
                  lhs,
                  1,
                  rhs,
                  -1,
                  output,
                  matmul,
                  outputTensors,
                  virtualGraphId,
                  pipelineStage,
                  executionPhase,
                  name);

  serializeVarUpdate(1,
                     builder,
                     output,
                     matmul,
                     outputTensors,
                     virtualGraphId,
                     pipelineStage,
                     executionPhase,
                     name);
}

static void
serializeFwdMatMul_OutputChannels(TransformBuilder &builder,
                                  Tensor *lhs,
                                  Tensor *rhs,
                                  Tensor *output,
                                  MatMulBaseOp *matmul,
                                  std::vector<TensorId> &outputTensors,
                                  OptionalVGraphId virtualGraphId,
                                  OptionalPipelineStage pipelineStage,
                                  OptionalExecutionPhase executionPhase,
                                  std::string name) {
  serializeMatMul(builder,
                  lhs,
                  -1,
                  rhs,
                  2,
                  output,
                  matmul,
                  outputTensors,
                  virtualGraphId,
                  pipelineStage,
                  executionPhase,
                  name);

  builder.concat(outputTensors,
                 2,
                 output->id,
                 virtualGraphId,
                 pipelineStage,
                 executionPhase,
                 name + "_Concat");

  builder.getGraph().topoCons->transfer(matmul, output->getProducer());
}

static void
serializeBwdLhsMatMul_OutputChannels(TransformBuilder &builder,
                                     Tensor *lhs,
                                     Tensor *rhs,
                                     Tensor *output,
                                     MatMulBaseOp *matmul,
                                     std::vector<TensorId> &outputTensors,
                                     OptionalVGraphId virtualGraphId,
                                     OptionalPipelineStage pipelineStage,
                                     OptionalExecutionPhase executionPhase,
                                     std::string name) {

  serializeMatMul(builder,
                  lhs,
                  2,
                  rhs,
                  1,
                  output,
                  matmul,
                  outputTensors,
                  virtualGraphId,
                  pipelineStage,
                  executionPhase,
                  name);

  sumByAddInplace(builder,
                  keepPrecision(matmul),
                  output,
                  outputTensors,
                  virtualGraphId,
                  pipelineStage,
                  executionPhase,
                  name);
}

static void
serializeBwdRhsMatMul_OutputChannels(TransformBuilder &builder,
                                     Tensor *lhs,
                                     Tensor *rhs,
                                     Tensor *output,
                                     MatMulBaseOp *matmul,
                                     std::vector<TensorId> &outputTensors,
                                     OptionalVGraphId virtualGraphId,
                                     OptionalPipelineStage pipelineStage,
                                     OptionalExecutionPhase executionPhase,
                                     std::string name) {
  serializeMatMul(builder,
                  lhs,
                  -1,
                  rhs,
                  2,
                  output,
                  matmul,
                  outputTensors,
                  virtualGraphId,
                  pipelineStage,
                  executionPhase,
                  name);

  serializeVarUpdate(2,
                     builder,
                     output,
                     matmul,
                     outputTensors,
                     virtualGraphId,
                     pipelineStage,
                     executionPhase,
                     name);
}

bool SerializeMatMuls::apply(Graph &graph) const {

  // Find all serialized mat muls
  auto matmuls = findSerializedMatMuls(graph);

  // Create the helper class
  TransformBuilder builder(graph);

  // So for each mat mul
  for (auto matmul : matmuls) {
    auto &matmulInputTensorMap  = matmul->input->tensorMap();
    auto &matmulOutputTensorMap = matmul->output->tensorMap();

    auto matmulLhs    = matmulInputTensorMap.at(MatMulOp::getLhsInIndex());
    auto matmulRhs    = matmulInputTensorMap.at(MatMulOp::getRhsInIndex());
    auto matmulOutput = matmulOutputTensorMap.at(MatMulOp::getOutIndex());
    auto outputShape  = matmulOutput->info.shape();
    auto factor       = matmul->getSerialiseSettings().factor;

    if (factor == 1) {
      continue;
    }

    logging::ir::info("matmul:{} {}x{}->{} mode: {} factor: {} phase: {}",
                      matmul->opid,
                      matmulLhs->info.shape(),
                      matmulRhs->info.shape(),
                      matmulOutput->info.shape(),
                      static_cast<int64_t>(matmul->getSerialiseSettings().mode),
                      factor,
                      static_cast<int64_t>(matmul->getPhase()));

    OptionalVGraphId virtualGraphId{};
    if (matmul->hasVirtualGraphId()) {
      virtualGraphId = matmul->getVirtualGraphId();
    }

    OptionalPipelineStage pipelineStage{};
    if (matmul->hasPipelineStage()) {
      pipelineStage = matmul->getPipelineStage();
    }

    OptionalExecutionPhase executionPhase{};
    if (matmul->hasExecutionPhase()) {
      executionPhase = matmul->getExecutionPhase();
    }

    std::string name = matmul->getName();

    matmul->disconnectAllInputs();
    matmul->disconnectAllOutputs();

    std::vector<TensorId> outputTensors;

    if (matmul->getSerialiseSettings().mode ==
        MatMulBaseOp::SerialiseSettings::Mode::InputChannels) {
      switch (matmul->getPhase()) {
      case MatMulOp::Phase::Fwd: {
        serializeFwdMatMul_InputChannels(builder,
                                         matmulLhs,
                                         matmulRhs,
                                         matmulOutput,
                                         matmul,
                                         outputTensors,
                                         virtualGraphId,
                                         pipelineStage,
                                         executionPhase,
                                         name);
      } break;
      case MatMulOp::Phase::BwdLHS: {
        serializeBwdLhsMatMul_InputChannels(builder,
                                            matmulLhs,
                                            matmulRhs,
                                            matmulOutput,
                                            matmul,
                                            outputTensors,
                                            virtualGraphId,
                                            pipelineStage,
                                            executionPhase,
                                            name);
      } break;
      case MatMulOp::Phase::BwdRHS: {
        serializeBwdRhsMatMul_InputChannels(builder,
                                            matmulLhs,
                                            matmulRhs,
                                            matmulOutput,
                                            matmul,
                                            outputTensors,
                                            virtualGraphId,
                                            pipelineStage,
                                            executionPhase,
                                            name);
      } break;
      };
    } else if (matmul->getSerialiseSettings().mode ==
               MatMulBaseOp::SerialiseSettings::Mode::OutputChannels) {
      switch (matmul->getPhase()) {
      case MatMulOp::Phase::Fwd: {
        serializeFwdMatMul_OutputChannels(builder,
                                          matmulLhs,
                                          matmulRhs,
                                          matmulOutput,
                                          matmul,
                                          outputTensors,
                                          virtualGraphId,
                                          pipelineStage,
                                          executionPhase,
                                          name);
      } break;
      case MatMulOp::Phase::BwdLHS: {
        serializeBwdLhsMatMul_OutputChannels(builder,
                                             matmulLhs,
                                             matmulRhs,
                                             matmulOutput,
                                             matmul,
                                             outputTensors,
                                             virtualGraphId,
                                             pipelineStage,
                                             executionPhase,
                                             name);
      } break;
      case MatMulOp::Phase::BwdRHS: {
        serializeBwdRhsMatMul_OutputChannels(builder,
                                             matmulLhs,
                                             matmulRhs,
                                             matmulOutput,
                                             matmul,
                                             outputTensors,
                                             virtualGraphId,
                                             pipelineStage,
                                             executionPhase,
                                             name);
      } break;
      };
    } else if (matmul->getSerialiseSettings().mode ==
               MatMulBaseOp::SerialiseSettings::Mode::ReducingDim) {
      switch (matmul->getPhase()) {
      case MatMulOp::Phase::Fwd: {
        serializeFwdMatMul_ReducingDim(builder,
                                       matmulLhs,
                                       matmulRhs,
                                       matmulOutput,
                                       matmul,
                                       outputTensors,
                                       virtualGraphId,
                                       pipelineStage,
                                       executionPhase,
                                       name);
      } break;
      case MatMulOp::Phase::BwdLHS: {
        serializeBwdLhsMatMul_ReducingDim(builder,
                                          matmulLhs,
                                          matmulRhs,
                                          matmulOutput,
                                          matmul,
                                          outputTensors,
                                          virtualGraphId,
                                          pipelineStage,
                                          executionPhase,
                                          name);
      } break;
      case MatMulOp::Phase::BwdRHS: {
        serializeBwdRhsMatMul_ReducingDim(builder,
                                          matmulLhs,
                                          matmulRhs,
                                          matmulOutput,
                                          matmul,
                                          outputTensors,
                                          virtualGraphId,
                                          pipelineStage,
                                          executionPhase,
                                          name);
      } break;
      };
    }

    // Remove the original matmul
    graph.eraseOp(matmul->id);

    if (outputShape != matmulOutput->info.shape()) {
      throw internal_error("Serialization of matmul({}, {}) has changed output "
                           "shape from {} to {}",
                           matmulLhs->str(),
                           matmulRhs->str(),
                           outputShape,
                           matmulOutput->info.shape());
    }
  }

  // Update the graph vertices
  graph.getIr().updateVertices();

  // Remove any dangling tensors
  graph.getTensors().removeIsolated(true);

  return true;
}

namespace {
bool init = Transform::registerTransform(new SerializeMatMuls);
}

} // namespace popart
