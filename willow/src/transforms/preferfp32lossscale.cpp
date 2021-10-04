// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <popart/ir.hpp>
#include <popart/op/nll.hpp>
#include <popart/op/softmax.hpp>
#include <popart/optimizer.hpp>
#include <popart/transforms/preferfp32lossscale.hpp>
#include <popart/util.hpp>

namespace popart {
namespace {

// To return true, the op's implementation must be able to handle mixed
// precision maths. We have no good way to know this programmatically at the
// point of running this pattern, so we hard code this information here.
bool isMixedPrecisionLossGradOp(Op *op) {
  // All NLL-like operations
  if (op->isConvertibleTo<NllGradOp>()) {
    return true;
  } else if (op->isConvertibleTo<SoftmaxGradDirectOp>()) {
    return true;
  } else if (op->isConvertibleTo<NlllWithSoftmaxGradDirectOp>()) {
    return true;
  }

  return false;
}

} // namespace

bool shouldApply(const Graph &graph) {
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

  // All consumers must be able to handle fp16 input gradients, and
  // fp32 activations
  for (Op *consumer : lossScaleTensor->consumers.getOps()) {
    if (!isMixedPrecisionLossGradOp(consumer)) {
      return false;
    }
  }

  // All MPLGOs must produce fp16 gradients
  for (Op *consumer : lossScaleTensor->consumers.getOps()) {
    for (Tensor *output : consumer->output->tensors()) {
      if (output->info.dataType() != DataType::FLOAT16) {
        return false;
      }
    }
  }

  return true;
}

bool PreferFp32LossScale::apply(Graph &graph) const {
  if (!shouldApply(graph)) {
    return true;
  }

  Tensor *lossScaleTensor = getLossScaleTensor(graph);

  // Set data type to float
  lossScaleTensor->info.set(DataType::FLOAT);

  // Reset the tensor data with float data
  if (lossScaleTensor->hasTensorData()) {
    float lossScale    = graph.getIr().getOptimizer().getFinalLossScalingVal();
    auto convertedData = convertFloatToDataType(DataType::FLOAT, lossScale);
    lossScaleTensor->tensorData()->resetDataWithNonMatchingSize(
        lossScaleTensor->info, convertedData);
  }

  // A sanity check. For each consumer of lossScaleTensor re-call its Op's
  // setup function, and confirm that output type is still same
  for (Op *consumer : lossScaleTensor->consumers.getOps()) {
    std::map<TensorId, DataType> oldDataTypes, newDataTypes;
    for (Tensor *output : consumer->output->tensors()) {
      oldDataTypes.emplace(output->id, output->info.dataType());
    }
    consumer->setup();
    for (Tensor *output : consumer->output->tensors()) {
      newDataTypes.emplace(output->id, output->info.dataType());
    }
    if (oldDataTypes != newDataTypes) {
      throw internal_error("PreferFp32LossScale: the transform has modified "
                           "the output data type(s) of Op {}",
                           consumer->str());
    }
  }

  return true;
}

std::size_t PreferFp32LossScale::id() {
  return typeid(PreferFp32LossScale).hash_code();
}

namespace {
bool init = Transform::registerTransform(new PreferFp32LossScale);
}

} // namespace popart
