
// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <transforms/autodiff/gradgrowerloss.hpp>

#include <memory>

#include <popart/graph.hpp>
#include <popart/op.hpp>
#include <popart/op/scale.hpp>
#include <popart/opmanager.hpp>
#include <popart/optimizer.hpp>
#include <popart/pbwrap.hpp>
#include <popart/sessionoptions.hpp>
#include <popart/tensor.hpp>
#include <popart/tensornames.hpp>

#include <poplar/Target.hpp>

namespace popart {

GradGrowerLoss::GradGrowerLoss(AutodiffIrInterface &dep)
    : GradGrowerLossInterface(), GradGrower(dep) {}

Op *GradGrowerLoss::growLossGradients() {

  TensorId gradStarterId = getGradId(dep.get().getFinalLossId());
  TensorInfo gradStarterInfo =
      dep.get().getTensors().get(dep.get().getFinalLossId())->info;

  // If our optimiser uses loss scaling we need to multiply our loss gradient by
  // the loss scale. If the loss scale is a constant then we can do this here to
  // avoid doing an additional operation.
  const auto &optimizer = dep.get().getOptimizer();

  if (optimizer.lossScaling().isConst()) {
    // By default this will be 1.0f.
    float lossScale = optimizer.lossScaling().val();

    if (dep.get().getSessionOptions().accumulationAndReplicationReductionType ==
        ReductionType::Mean) {
      lossScale /= dep.get().getSessionOptions().getGlobalReplicationFactor();
    }

    switch (gradStarterInfo.dataType()) {
    case DataType::FLOAT: {
      std::vector<float> gradStarterData(1, lossScale);
      dep.get().getTensors().addConstInit(
          gradStarterId,
          gradStarterInfo,
          reinterpret_cast<void *>(gradStarterData.data()));
      break;
    }
    case DataType::FLOAT16: {
      std::vector<float> floatData(1, lossScale);
      std::vector<char> gradStarterData(2);
      poplar::copyFloatToDeviceHalf(
          poplar::Target(), floatData.data(), gradStarterData.data(), 1);
      dep.get().getTensors().addConstInit(
          gradStarterId,
          gradStarterInfo,
          reinterpret_cast<void *>(gradStarterData.data()));
      break;
    }
    case DataType::INT16: {
      std::vector<int16_t> gradStarterData(1, static_cast<int16_t>(lossScale));
      dep.get().getTensors().addConstInit(
          gradStarterId,
          gradStarterInfo,
          reinterpret_cast<void *>(gradStarterData.data()));
      break;
    }
    case DataType::INT32: {
      std::vector<int32_t> gradStarterData(1, static_cast<int32_t>(lossScale));
      dep.get().getTensors().addConstInit(
          gradStarterId,
          gradStarterInfo,
          reinterpret_cast<void *>(gradStarterData.data()));
      break;
    }
    case DataType::INT64: {
      std::vector<int64_t> gradStarterData(1, static_cast<int64_t>(lossScale));
      dep.get().getTensors().addConstInit(
          gradStarterId,
          gradStarterInfo,
          reinterpret_cast<void *>(gradStarterData.data()));
      break;
    }
    case DataType::UINT32: {
      std::vector<uint32_t> gradStarterData(1,
                                            static_cast<uint32_t>(lossScale));
      dep.get().getTensors().addConstInit(
          gradStarterId,
          gradStarterInfo,
          reinterpret_cast<void *>(gradStarterData.data()));
      break;
    }
    case DataType::UINT64: {
      std::vector<uint64_t> gradStarterData(1,
                                            static_cast<uint64_t>(lossScale));
      dep.get().getTensors().addConstInit(
          gradStarterId,
          gradStarterInfo,
          reinterpret_cast<void *>(gradStarterData.data()));
      break;
    }
    // Making it explicit which data types we're not handling. Note that
    // the logic will fall through to the error.
    case DataType::UINT8:
    case DataType::INT8:
    case DataType::UINT16:
    case DataType::BOOL:
    case DataType::BFLOAT16:
    case DataType::DOUBLE:
    case DataType::COMPLEX64:
    case DataType::COMPLEX128:
    case DataType::STRING:
    case DataType::UNDEFINED:
    default: {
      throw error("Unexpected loss data-type, '{}'",
                  gradStarterInfo.getDataTypeInfo()->name());
    }
    }

    return nullptr;
  } else {
    // In the case where the user wants to apply loss scaling with a scaling
    // factor that is not constant we need to apply scaling differently. We need
    // the finalLossOp gradient tensor to match the optimizer's loss scaling
    // tensor.
    TensorId lossScalingId =
        optimizer.getLossScalingTensorId(gradStarterInfo.dataType());
    std::unique_ptr<popart::Op> lossScalingInputOp = OpManager::createOp(
        Domain::ai_graphcore,
        "Scale",
        dep.get().getOpSetVersionFromModel(Domain::ai_graphcore),
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
    if (dep.get().getSessionOptions().accumulationAndReplicationReductionType ==
        ReductionType::Mean) {
      dynamic_cast<ScaleOp *>(op)->setScaleFactor(
          1.0f / dep.get().getSessionOptions().getGlobalReplicationFactor());
    } else {
      dynamic_cast<ScaleOp *>(op)->setScaleFactor(1.0f);
    }
    op->setup();
    return op;
  }
}

} // namespace popart
