
// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <functional>
#include <memory>
#include <string>
#include <transforms/autodiff/gradgrowerloss.hpp>
#include <utility>
#include <vector>
#include <popart/graph.hpp>
#include <popart/op.hpp>
#include <popart/opmanager.hpp>
#include <popart/optimizer.hpp>
#include <popart/pbwrap.hpp>
#include <popart/tensor.hpp>
#include <popart/tensornames.hpp>

#include "popart/names.hpp"
#include "popart/operatoridentifier.hpp"
#include "popart/optimizervalue.hpp"
#include "popart/tensordebuginfo.hpp"
#include "popart/tensorinfo.hpp"
#include "popart/tensors.hpp"
#include "transforms/autodiff/autodiffhelper.hpp"
#include "transforms/autodiff/autodiffirinterface.hpp"

namespace popart {

GradGrowerLoss::GradGrowerLoss(AutodiffIrInterface &dep)
    : GradGrowerLossInterface(), AutodiffHelper(dep) {}

Op *GradGrowerLoss::growLossGradients() {

  TensorId gradStarterId = getGradId(dep.get().getFinalLossId());
  TensorInfo gradStarterInfo =
      dep.get().getTensors().get(dep.get().getFinalLossId())->info;

  // If our optimiser uses loss scaling we need to multiply our loss gradient by
  // the loss scale. If the loss scale is a constant then we can do this here to
  // avoid doing an additional operation.
  const auto &optimizer = dep.get().getOptimizer();

  if (optimizer.lossScaling().isConst()) {
    float lossScale = optimizer.getFinalLossScalingVal();

    addConstInitFromFloat(
        lossScale, gradStarterId, gradStarterInfo, dep.get().getTensors());

    return nullptr;
  } else {
    // Connect the streamed loss scale tensor to the gradient of the
    // final loss tensor via an identity op.
    TensorId lossScalingId =
        optimizer.getLossScalingTensorId(gradStarterInfo.dataType());
    std::unique_ptr<popart::Op> lossScalingInputOp =
        OpManager::createOp(Domain::ai_onnx,
                            "Identity",
                            dep.get().getOpSetVersionFromModel(Domain::ai_onnx),
                            dep.get().getMainGraph());

    OpId lossScalingInputOpId =
        dep.get().getMainGraph().moveIntoGraph(std::move(lossScalingInputOp));

    std::vector<TensorId> inputs{lossScalingId};
    std::vector<TensorId> outputs{getGradId(dep.get().getFinalLossId())};
    dep.get().getMainGraph().connectInputs(InputVecWrapper(inputs),
                                           lossScalingInputOpId);
    dep.get().getMainGraph().connectOutputs(OutputVecWrapper(outputs),
                                            lossScalingInputOpId);
    Op *op = dep.get().getMainGraph().getOp(lossScalingInputOpId);
    op->setup();
    return op;
  }
}

} // namespace popart
