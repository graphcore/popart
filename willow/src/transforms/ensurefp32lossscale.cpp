// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <cstddef>
#include <map>
#include <memory>
#include <ostream>
#include <string>
#include <typeinfo>
#include <utility>
#include <vector>
#include <popart/alias/aliasmodel.hpp>
#include <popart/alias/aliasmodelgrower.hpp>
#include <popart/graphutils.hpp>
#include <popart/ir.hpp>
#include <popart/op/cast.hpp>
#include <popart/op/l1.hpp>
#include <popart/op/nll.hpp>
#include <popart/op/softmax.hpp>
#include <popart/optimizer.hpp>
#include <popart/transforms/ensurefp32lossscale.hpp>
#include <popart/util.hpp>

#include "popart/datatype.hpp"
#include "popart/error.hpp"
#include "popart/graph.hpp"
#include "popart/graphid.hpp"
#include "popart/logging.hpp"
#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/operators.hpp"
#include "popart/tensor.hpp"
#include "popart/tensordata.hpp"
#include "popart/tensordebuginfo.hpp"
#include "popart/tensorindex.hpp"
#include "popart/tensorinfo.hpp"
#include "popart/transforms/transform.hpp"

namespace popart {

bool EnsureFp32LossScale::isMixedPrecisionLossGradOp(Op *op) {
  // All NllLoss-grad-like operations
  if (op->isConvertibleTo<NllGradOp>()) {
    return true;
  } else if (op->isConvertibleTo<SoftmaxGradDirectOp>()) {
    return true;
  } else if (op->isConvertibleTo<NlllWithSoftmaxGradDirectOp>()) {
    return true;
  }

  // L1Grad Op
  else if (op->isConvertibleTo<L1GradOp>()) {
    return true;
  }

  return false;
}

Tensor *EnsureFp32LossScale::getLossScaleInputTensor(Op *op) {
  // All NllLoss-grad-like operations
  if (op->isConvertibleTo<NllGradOp>()) {
    return op->inTensor(NllGradOp::getGradInIndex());
  } else if (op->isConvertibleTo<SoftmaxGradDirectOp>()) {
    return op->inTensor(SoftmaxGradDirectOp::getGradProbsInIndex());
  } else if (op->isConvertibleTo<NlllWithSoftmaxGradDirectOp>()) {
    return op->inTensor(NlllWithSoftmaxGradDirectOp::getGradProbsInIndex());
  }

  // L1Grad Op
  else if (op->isConvertibleTo<L1GradOp>()) {
    return op->inTensor(L1GradOp::getGradInIndex());
  }

  throw internal_error("EnsureFp32LossScale Transform: unexpected op type.");
}

bool EnsureFp32LossScale::isPassThroughOp(Op *op) const {
  // Single input op
  if (op->input->n() == 1) {

    // All outputs must have same type as input
    DataType inputType = op->input->tensors().front()->info.dataType();
    for (Tensor *output : op->output->tensors()) {
      if (output->info.dataType() != inputType) {
        return false;
      }
    }
    return true;
  }

  return false;
}

FromLossScaleTraversalOps
EnsureFp32LossScale::traverseFromLossScaleTensor(const Graph &graph) const {
  Tensor *lossScaleTensor = getLossScaleTensor(graph);

  std::vector<Op *> passThroughOps, mplgoCandidates;

  const auto &graphId = graph.id;

  auto visitor = [this, &passThroughOps, &mplgoCandidates, &graphId](
                     Tensor *t) {
    if (!(t->hasProducer())) {
      return true;
    }

    Op *producer = t->getProducer();

    // Pass-through ops. Continue traversal
    if (isPassThroughOp(producer)) {
      passThroughOps.push_back(producer);
      return true;
    }

    // MPLGO candidates. Terminate traversal here
    else {
      if (std::find(mplgoCandidates.begin(), mplgoCandidates.end(), producer) ==
          mplgoCandidates.end()) {
        // All valid MPLGO candidates must produce fp16 gradients
        for (Tensor *output : producer->output->tensors()) {
          if (output->info.dataType() != DataType::FLOAT16) {
            throw error("EnsureFp32LossScale: Unable to apply the transform on "
                        "graph '{}'. Graph traversal terminating at Op '{}', "
                        "but output tensor '{}' is not of data type fp16.",
                        graphId,
                        producer->str(),
                        output->id);
          }
        }

        mplgoCandidates.push_back(producer);
      }
      return false;
    }
  };

  graphutils::traverse(
      {lossScaleTensor},
      visitor,
      [](Op *op, Tensor *tq, Tensor *tn) { return true; }, // no filter
      graphutils::TraversalType::BreadthFirst,
      graphutils::VisitType::Pre,
      graphutils::TraversalDirection::Forward);

  return {passThroughOps, mplgoCandidates};
}

bool EnsureFp32LossScale::shouldApply(const Graph &graph) const {
  // Only relevant if we are training
  if (!graph.getIr().canTrain()) {
    return false;
  }

  if (!hasSingleConnectedLossScaleTensor(graph)) {
    return false;
  }
  Tensor *lossScaleTensor = getLossScaleTensor(graph);

  // Requires loss scale tensor is fp16
  if (lossScaleTensor->info.dataType() != DataType::FLOAT16) {
    return false;
  }

  return true;
}

bool hasFp32Fp16MixedPrecisionInputs(Op *op) {
  bool hasFp16Inputs = false;
  bool hasFp32Inputs = false;
  for (Tensor *input : op->input->tensors()) {
    if (input->info.dataType() == DataType::FLOAT16) {
      hasFp16Inputs = true;
    } else if (input->info.dataType() == DataType::FLOAT) {
      hasFp32Inputs = true;
    }
  }

  return hasFp16Inputs && hasFp32Inputs;
}

/**
 * Before:
 *   tensor -- Op
 *        '--- AnotherOp
 * After:
 *   tensor -- Cast(fp32) -- upCastTensor -- Op
 *        '--- AnotherOp
 **/
void EnsureFp32LossScale::upCastTensor(Op *op, InIndex index) const {
  Tensor *tensor = op->inTensor(index);
  if (tensor->info.dataType() != DataType::FLOAT16) {
    throw error("EnsureFp32LossScale: Cannot upcast tensor '{}', as it does "
                "not have data type 'DataType::FLOAT16'",
                tensor->id);
  }

  op->disconnectInTensor(index, tensor);

  TensorId upCastTensorId = op->getIr().createIntermediateTensorId(tensor->id);
  auto &g                 = op->getGraph();
  auto castOp =
      g.createConnectedOp<CastOp>({{CastOp::getInIndex(), tensor->id}},
                                  {{CastOp::getOutIndex(), upCastTensorId}},
                                  Onnx::Operators::Cast_9,
                                  DataType::FLOAT,
                                  Op::Settings(g, ""));

  op->connectInTensor(index, upCastTensorId);

  AliasModel aliasModel;
  AliasModelGrower aliasModelGrower{aliasModel};
  aliasModelGrower.growFullGraph(g, DataDependenciesOnly::Yes);
  castOp->inheritPlacementAttributes(false, aliasModel);
}

/**
 * Before:
 *   Op -- tensor -- AnotherOp
 * After:
 *   Op -- tensor -- CastOp -- downCastTensor -- AnotherOp
 **/
void EnsureFp32LossScale::downCastTensor(Tensor *tensor) const {
  if (tensor->info.dataType() != DataType::FLOAT) {
    throw error("EnsureFp32LossScale: Cannot downcast tensor '{}', as it does "
                "not have data type 'DataType::FLOAT32'",
                tensor->id);
  }

  TensorId downCastTensorId =
      tensor->getIr().createIntermediateTensorId(tensor->id);
  auto &g = tensor->getGraph();
  auto castOp =
      g.createConnectedOp<CastOp>({{CastOp::getInIndex(), tensor->id}},
                                  {{CastOp::getOutIndex(), downCastTensorId}},
                                  Onnx::Operators::Cast_9,
                                  DataType::FLOAT16,
                                  Op::Settings(g, ""));
  AliasModel aliasModel;
  AliasModelGrower aliasModelGrower{aliasModel};
  aliasModelGrower.growFullGraph(g, DataDependenciesOnly::Yes);
  castOp->inheritPlacementAttributes(false, aliasModel);

  // Disconnect tensor's consumers, and reconnect to output of castOp
  for (Op *op : tensor->consumers.getOps()) {
    if (op != castOp) {
      for (InIndex index : op->input->indices(tensor)) {
        op->disconnectInTensor(index, tensor);
        op->connectInTensor(index, downCastTensorId);
      }
    }
  }
}

/**
 * Apply as follows:
 * 1. Convert loss scale tensor to fp32
 * 2. For each passThroughOp re-call setup() such that output tensors are now
 *    fp32
 * 3. For each mplgoCandidate:
 *   3.1. If candidate is valid MPLGO, check output types are unchanged
 *   3.2. If candidate is not a valid MPLGO, upcast fp16 inputs to fp32,
 *        re-call setup(), check outputs are fp32, downcast outputs to fp16
 **/
bool EnsureFp32LossScale::apply(Graph &graph) const {
  if (!shouldApply(graph)) {
    return true;
  }

  auto traversalResults             = traverseFromLossScaleTensor(graph);
  std::vector<Op *> passThroughOps  = traversalResults.first;
  std::vector<Op *> mplgoCandidates = traversalResults.second;

  // 1. Set loss scale data type to float
  Tensor *lossScaleTensor = getLossScaleTensor(graph);
  lossScaleTensor->info.set(DataType::FLOAT);

  // Reset the tensor data with float data
  if (lossScaleTensor->hasTensorData()) {
    float lossScale    = graph.getIr().getOptimizer().getFinalLossScalingVal();
    auto convertedData = convertFloatToDataType(DataType::FLOAT, lossScale);
    lossScaleTensor->tensorData()->resetDataWithNonMatchingSize(
        lossScaleTensor->info, convertedData);
  }

  // 2. Re-call Op setup functions to propagate the loss scale data type change
  // to the MPLGO inputs
  auto tensorTypeCheck =
      [](Op *op, Tensor *t, DataType dataType, const std::string &direction) {
        if (t->info.dataType() != dataType) {
          std::stringstream ss;
          ss << "EnsureFp32LossScale: Pass-through Op " << op->str();
          ss << " has non-" << dataType << " " << direction << " tensor "
             << t->id;

          throw internal_error(ss.str());
        }
        return;
      };

  for (Op *passThroughOp : passThroughOps) {
    // Confirm input is fp32
    for (Tensor *input : passThroughOp->input->tensors()) {
      tensorTypeCheck(passThroughOp, input, DataType::FLOAT, "input");
    }

    // Confirm output is fp16
    for (Tensor *output : passThroughOp->output->tensors()) {
      tensorTypeCheck(passThroughOp, output, DataType::FLOAT16, "output");
    }

    passThroughOp->setup();

    // Re-call setup function and confirm output is now fp32
    for (Tensor *output : passThroughOp->output->tensors()) {
      tensorTypeCheck(passThroughOp, output, DataType::FLOAT, "output");
    }
  }

  // 3.
  for (Op *candidate : mplgoCandidates) {
    std::map<TensorId, DataType> oldDataTypes;
    for (Tensor *output : candidate->output->tensors()) {
      oldDataTypes.emplace(output->id, output->info.dataType());
    }

    // A valid candidate must:
    //   - be able to handle fp16 input gradients, and fp32 activations
    //   - have a scalar loss scale gradient input tensor
    if (isMixedPrecisionLossGradOp(candidate) &&
        getLossScaleInputTensor(candidate)->info.nelms() == 1) {
      // 3.1. Re-call candidate's setup() function, and check outputs
      //      types are unchanged
      candidate->setup();

      std::map<TensorId, DataType> newDataTypes;
      for (Tensor *output : candidate->output->tensors()) {
        newDataTypes.emplace(output->id, output->info.dataType());
      }

      if (oldDataTypes != newDataTypes) {
        throw internal_error("EnsureFp32LossScale: the transform has modified "
                             "the output data type(s) of Op {}",
                             candidate->str());
      }
    }

    else {
      // 3.2

      // Verify candidate has mixed precision inputs
      if (!hasFp32Fp16MixedPrecisionInputs(candidate)) {
        throw internal_error("EnsureFp32LossScale: MPLGO candidate '{}' "
                             "expected to have mixed-precision inputs",
                             candidate->str());
      }

      // Upcast fp16 inputs to fp32
      for (int i = 0; i < candidate->input->n(); i++) {
        Tensor *input = candidate->input->tensor(i);
        if (input->info.dataType() == DataType::FLOAT16) {
          upCastTensor(candidate, i);
        }
      }

      // Re-call setup
      candidate->setup();

      // Check all fp16 outputs are now fp32
      for (const auto &id_type : oldDataTypes) {
        TensorId id          = id_type.first;
        DataType oldDataType = id_type.second;
        if (oldDataType == DataType::FLOAT16) {
          if (graph.getTensor(id)->info.dataType() == DataType::FLOAT) {
            // Downcast fp32 outputs that were previously fp16
            downCastTensor(graph.getTensor(id));
          } else {
            throw internal_error(
                "EnsureFp32LossScale: expecting data type of tensor '{}' to "
                "have been converted to fp32 by this transform.",
                id);
          }
        }
      }
    }
  }

  return true;
}

std::size_t EnsureFp32LossScale::id() {
  return typeid(EnsureFp32LossScale).hash_code();
}

namespace {
bool init = Transform::registerTransform(new EnsureFp32LossScale);
}

} // namespace popart
