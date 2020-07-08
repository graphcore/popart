// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <memory>
#include <popart/ces/concatce.hpp>
#include <popart/ces/flattence.hpp>
#include <popart/ces/slicece.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/onnxutil.hpp>
#include <popart/op/accumulate.hpp>
#include <popart/op/accumulatorupdate.hpp>
#include <popart/op/adamcombo.hpp>
#include <popart/op/adamupdater.hpp>
#include <popart/op/adamvarupdate.hpp>
#include <popart/op/collectives/replicatedallreduce.hpp>
#include <popart/op/concat.hpp>
#include <popart/op/flatten.hpp>
#include <popart/op/lamb.hpp>
#include <popart/op/slice.hpp>
#include <popart/patterns/adamdecompose.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/topocons.hpp>

namespace popart {

bool AdamDecompose::matches(Op *op) const {
  return op->isConvertibleTo<AdamComboOp>();
}

std::vector<const Tensor *> AdamDecompose::touches(Op *) const { return {}; }

namespace {
template <typename T>
void addAdamStateTensor(AdamComboOp &op,
                        const TensorId &tensorId,
                        const TensorInfo info) {
  auto &graph = op.getGraph();
  auto &ir    = graph.getIr();
  if (ir.tensorExistsInInitialisers(tensorId)) {
    auto tp = onnxutil::getTensorProto(ir.getModel(), tensorId);
    graph.getTensors().addVarInit(tensorId, &tp);
  } else {
    std::vector<T> d(info.nelms(), static_cast<T>(0));
    graph.getTensors().addVarInit(tensorId, info, d.data());
  }
}
} // namespace

bool AdamDecompose::apply(Op *op) const {

  auto &ir    = op->getIr();
  auto &graph = op->getGraph();

  auto storeTensor = [&ir](TensorId id) {
    if (ir.additionalModelProtoTensors.find(id) ==
            ir.additionalModelProtoTensors.end() &&
        !ir.tensorExistsInInitialisers(id)) {
      // If we are not going to stream the tensors from the host,
      // don't add them to the set of additional tensors to be saved
      // in the onnx modelproto
      if (!ir.storingIsDisabledForTensor(id)) {
        ir.additionalModelProtoTensors.insert(id);
      }
    }
  };

  // matches must have verified the correctness before this call
  auto combo = static_cast<AdamComboOp *>(op);

  Tensor *weightGrad =
      combo->inTensor(VarUpdateWithUpdaterOp::getUpdaterInIndex());
  Tensor *weight    = combo->inTensor(VarUpdateOp::getVarToUpdateInIndex());
  Tensor *newWeight = combo->outTensor(VarUpdateOp::getUpdatedVarOutIndex());

  TensorId weightGradId    = weightGrad->id;
  TensorId weightId        = weight->id;
  TensorId updatedWeightId = newWeight->id;

  // Qualified tensor names for key tensors of the Adam/Lamb optimizers
  auto stepId        = reservedStepPrefix() + weightId;
  auto accumId       = reservedAccumPrefix() + weightId;
  auto accl1Id       = reservedAccl1Prefix() + weightId;
  auto accl2Id       = reservedAccl2Prefix() + weightId;
  auto adamUpdaterId = reservedAdamUpdaterPrefix() + weightId;
  auto lambR1SqId    = reservedLambR1SqPrefix() + weightId;
  auto lambR2SqId    = reservedLambR2SqPrefix() + weightId;

  // Step
  addAdamStateTensor<uint32_t>(*combo, stepId, TensorInfo(DataType::FLOAT, {}));
  storeTensor(stepId);

  if (weightGrad->info.dataType() != weight->info.dataType()) {
    throw error("Currently, weight and weight gradient should have the same "
                "type in AdamDecompose, this is outstanding work");
  }

  auto weightInfo  = weight->info;
  auto weightShape = weightInfo.shape();

  // Accumulator
  if (combo->withGradAccum) {
    auto accumInfo = TensorInfo(combo->accumType, weightShape);
    switch (combo->accumType) {
    case DataType::FLOAT:
      addAdamStateTensor<float>(*combo, accumId, accumInfo);
      break;
    case DataType::FLOAT16:
      addAdamStateTensor<float16_t>(*combo, accumId, accumInfo);
      break;
    default:
      error("Unsupported data type for tensor {}, "
            "currently only FLOAT16 and FLOAT are supported",
            accumId);
    }
  }

  // 1st momentum (accl1)
  auto accl1Info = TensorInfo(combo->accl1Type, weightShape);
  switch (combo->accl1Type) {
  case DataType::FLOAT:
    addAdamStateTensor<float>(*combo, accl1Id, accl1Info);
    break;
  case DataType::FLOAT16:
    addAdamStateTensor<float16_t>(*combo, accl1Id, accl1Info);
    break;
  default:
    error("Unsupported data type for tensor {}, "
          "currently only FLOAT16 and FLOAT are supported",
          accl1Id);
  }

  // 2nd momentum (accl2)
  auto accl2Info = TensorInfo(combo->accl2Type, weightShape);
  switch (combo->accl2Type) {
  case DataType::FLOAT:
    addAdamStateTensor<float>(*combo, accl2Id, accl2Info);
    break;
  case DataType::FLOAT16:
    addAdamStateTensor<float16_t>(*combo, accl2Id, accl2Info);
    break;
  default:
    error("Unsupported data type for tensor {}, "
          "currently only FLOAT16 and FLOAT are supported",
          accl2Id);
  }

  TensorId gradIntoAcclId  = weightGradId;
  TensorId gradIntoAccumId = weightGradId;

  if (combo->reductionType == OptimizerReductionType::GradReduce) {
    auto reduceOpUp = std::make_unique<ReplicatedAllReduceOp>(
        Onnx::CustomOperators::ReplicatedAllReduce,
        Op::Settings(graph, combo->name() + "_reduce"));
    auto reduceOp = reduceOpUp.get();
    transferBaseProperties(combo, reduceOp);
    graph.moveIntoGraph(std::move(reduceOpUp));

    logging::pattern::trace("Connecting input {} to {} at {}",
                            weightGradId,
                            reduceOp->str(),
                            ReplicatedAllReduceOp::getInIndex());
    reduceOp->connectInTensor(ReplicatedAllReduceOp::getInIndex(),
                              weightGradId);

    // The reduced gradient
    gradIntoAcclId  = ir.createIntermediateTensorId(weightGradId);
    gradIntoAccumId = gradIntoAcclId;

    reduceOp->createAndConnectOutTensor(ReplicatedAllReduceOp::getOutIndex(),
                                        gradIntoAcclId);

    reduceOp->setup();
  }

  // Gradient accumulation
  if (combo->withGradAccum) {
    auto accumOpUp = std::make_unique<AccumulateOp>(
        accumId,
        AccumulationType::Add,
        OptimizerValue(1.0f),
        Op::Settings(graph, combo->name() + "_accumulate"));
    auto accumOp = accumOpUp.get();
    transferBaseProperties(combo, accumOp);
    graph.moveIntoGraph(std::move(accumOpUp));

    logging::pattern::trace("Connecting input {} to {} at {}",
                            accumId,
                            accumOp->str(),
                            VarUpdateOp::getVarToUpdateInIndex());
    accumOp->connectInTensor(VarUpdateOp::getVarToUpdateInIndex(), accumId);

    logging::pattern::trace("Connecting input {} to {} at {}",
                            gradIntoAccumId,
                            accumOp->str(),
                            VarUpdateWithUpdaterOp::getUpdaterInIndex());
    accumOp->connectInTensor(VarUpdateWithUpdaterOp::getUpdaterInIndex(),
                             gradIntoAccumId);

    // The updated accumulator
    TensorId updatedAccumId = ir.createIntermediateTensorId(accumId);

    accumOp->createAndConnectOutTensor(VarUpdateOp::getUpdatedVarOutIndex(),
                                       updatedAccumId);

    accumOp->setup();
    storeTensor(accumId);

    if (combo->reductionType == OptimizerReductionType::AccumReduce) {
      auto reduceOpUp = std::make_unique<ReplicatedAllReduceInplaceOp>(
          Onnx::CustomOperators::ReplicatedAllReduceInplace,
          Op::Settings(graph, combo->name() + "_reduce"));
      auto reduceOp = reduceOpUp.get();
      transferBaseProperties(combo, reduceOp);
      graph.moveIntoGraph(std::move(reduceOpUp));

      logging::pattern::trace("Connecting input {} to {} at {}",
                              updatedAccumId,
                              reduceOp->str(),
                              ReplicatedAllReduceInplaceOp::getInIndex());
      reduceOp->connectInTensor(ReplicatedAllReduceInplaceOp::getInIndex(),
                                updatedAccumId);

      // The updated, reduced accumulator
      gradIntoAcclId = ir.createIntermediateTensorId(accumId);

      reduceOp->createAndConnectOutTensor(
          ReplicatedAllReduceInplaceOp::getOutIndex(), gradIntoAcclId);

      reduceOp->setup();
      if (combo->withGradAccum) {
        reduceOp->settings.executionContext =
            ExecutionContext::AccumulateOuterFragment;
      }
    } else {
      // No replicated accumulator reduction
      gradIntoAcclId = updatedAccumId;
    }
  }

  // 1st momentum
  auto accl1OpUp = std::make_unique<AccumulateOp>(
      accl1Id,
      AccumulationType::MovingAverage,
      combo->initB1,
      Op::Settings(graph, combo->name() + "_accl1"));
  auto accl1Op = accl1OpUp.get();
  transferBaseProperties(combo, accl1Op);
  graph.moveIntoGraph(std::move(accl1OpUp));

  logging::pattern::trace("Connecting input {} to {} at {}",
                          accl1Id,
                          accl1Op->str(),
                          VarUpdateOp::getVarToUpdateInIndex());
  accl1Op->connectInTensor(VarUpdateOp::getVarToUpdateInIndex(), accl1Id);

  logging::pattern::trace("Connecting input {} to {} at {}",
                          gradIntoAcclId,
                          accl1Op->str(),
                          VarUpdateWithUpdaterOp::getUpdaterInIndex());
  accl1Op->connectInTensor(VarUpdateWithUpdaterOp::getUpdaterInIndex(),
                           gradIntoAcclId);

  // The updated accl1
  TensorId updatedAccl1Id = ir.createIntermediateTensorId(accl1Id);

  accl1Op->createAndConnectOutTensor(VarUpdateOp::getUpdatedVarOutIndex(),
                                     updatedAccl1Id);

  if (!combo->initB1.isConst()) {
    accl1Op->connectInTensor(AccumulateOp::getFactorInIndex(),
                             combo->inId(AdamComboOp::getBeta1InIndex()));
  }

  accl1Op->setup();
  if (combo->withGradAccum) {
    accl1Op->settings.executionContext =
        ExecutionContext::AccumulateOuterFragment;
  }
  storeTensor(accl1Id);

  // 2nd momentum
  auto accl2OpUp = std::make_unique<AccumulateOp>(
      accl2Id,
      AccumulationType::MovingAverageSquare,
      combo->initB2,
      Op::Settings(graph, combo->name() + "_accl2"));
  auto accl2Op = accl2OpUp.get();
  transferBaseProperties(combo, accl2Op);
  graph.moveIntoGraph(std::move(accl2OpUp));

  logging::pattern::trace("Connecting input {} to {} at {}",
                          accl2Id,
                          accl2Op->str(),
                          VarUpdateOp::getVarToUpdateInIndex());
  accl2Op->connectInTensor(VarUpdateOp::getVarToUpdateInIndex(), accl2Id);

  logging::pattern::trace("Connecting input {} to {} at {}",
                          gradIntoAcclId,
                          accl2Op->str(),
                          VarUpdateWithUpdaterOp::getUpdaterInIndex());
  accl2Op->connectInTensor(VarUpdateWithUpdaterOp::getUpdaterInIndex(),
                           gradIntoAcclId);

  // The updated accl2
  TensorId updatedAccl2Id = ir.createIntermediateTensorId(accl2Id);

  accl2Op->createAndConnectOutTensor(VarUpdateOp::getUpdatedVarOutIndex(),
                                     updatedAccl2Id);

  if (!combo->initB2.isConst()) {
    accl2Op->connectInTensor(AccumulateOp::getFactorInIndex(),
                             combo->inId(AdamComboOp::getBeta2InIndex()));
  }

  accl2Op->setup();
  if (combo->withGradAccum) {
    accl2Op->settings.executionContext =
        ExecutionContext::AccumulateOuterFragment;
  }
  storeTensor(accl2Id);

  // The accumulator updater
  if (combo->withGradAccum) {
    auto accumUpdateOpUp = std::make_unique<AccumulatorUpdateOp>(
        accumId, Op::Settings(graph, combo->name() + "_accumupdate"));
    auto accumUpdateOp = accumUpdateOpUp.get();
    transferBaseProperties(combo, accumUpdateOp);
    graph.moveIntoGraph(std::move(accumUpdateOpUp));

    logging::pattern::trace("Connecting input {} to {} at {}",
                            accumId,
                            accumUpdateOp->str(),
                            LambSquareOp::getInIndex());
    accumUpdateOp->connectInTensor(AccumulatorUpdateOp::getVarToUpdateInIndex(),
                                   accumId);

    TensorId updatedAccumId = reservedUpdatedVarPrefix() + accumId;

    accumUpdateOp->createAndConnectOutTensor(LambSquareOp::getOutIndex(),
                                             updatedAccumId);

    accumUpdateOp->setup();

    if (combo->withGradAccum) {
      accumUpdateOp->settings.executionContext =
          ExecutionContext::AccumulateOuterFragment;
    }

    // Accumulator update after accl1 and accl2
    graph.topoCons->insert(accl1Op, accumUpdateOp);
    graph.topoCons->insert(accl2Op, accumUpdateOp);
  }

  // Adam updater term
  auto adamUpdOpUp = std::make_unique<AdamUpdaterOp>(
      combo->mode,
      combo->initWd,
      combo->initB1,
      combo->initB2,
      combo->initEps,
      combo->initLs,
      Op::Settings(graph, combo->name() + "_adamupdater"));
  auto adamUpdOp = adamUpdOpUp.get();
  transferBaseProperties(combo, adamUpdOp);
  graph.moveIntoGraph(std::move(adamUpdOpUp));

  // Weight
  logging::pattern::trace("Connecting input {} to {} at {}",
                          weightId,
                          adamUpdOp->str(),
                          AdamUpdaterOp::getVarInIndex());
  adamUpdOp->connectInTensor(AdamUpdaterOp::getVarInIndex(), weightId);

  // 1st momentum
  logging::pattern::trace("Connecting input {} to {} at {}",
                          accl1Id,
                          adamUpdOp->str(),
                          AdamUpdaterOp::getAccl1InIndex());
  adamUpdOp->connectInTensor(AdamUpdaterOp::getAccl1InIndex(), updatedAccl1Id);

  // 2nd momentum
  logging::pattern::trace("Connecting input {} to {} at {}",
                          accl2Id,
                          adamUpdOp->str(),
                          AdamUpdaterOp::getAccl2InIndex());
  adamUpdOp->connectInTensor(AdamUpdaterOp::getAccl2InIndex(), updatedAccl2Id);

  // step
  logging::pattern::trace("Connecting input {} to {} at {}",
                          stepId,
                          adamUpdOp->str(),
                          AdamUpdaterOp::getStepInIndex());
  adamUpdOp->connectInTensor(AdamUpdaterOp::getStepInIndex(), stepId);

  // Optimizer parameters
  if (!combo->initWd.isConst()) {
    adamUpdOp->connectInTensor(AdamUpdaterOp::getWdInIndex(),
                               combo->inId(AdamComboOp::getWdInIndex()));
  }
  if (!combo->initB1.isConst()) {
    adamUpdOp->connectInTensor(AdamUpdaterOp::getBeta1InIndex(),
                               combo->inId(AdamComboOp::getBeta1InIndex()));
  }
  if (!combo->initB2.isConst()) {
    adamUpdOp->connectInTensor(AdamUpdaterOp::getBeta2InIndex(),
                               combo->inId(AdamComboOp::getBeta2InIndex()));
  }
  if (!combo->initEps.isConst()) {
    adamUpdOp->connectInTensor(AdamUpdaterOp::getEpsInIndex(),
                               combo->inId(AdamComboOp::getEpsInIndex()));
  }
  if (!combo->initLs.isConst()) {
    adamUpdOp->connectInTensor(AdamUpdaterOp::getLsInIndex(),
                               combo->inId(AdamComboOp::getLsInIndex()));
  }

  // Updater term
  adamUpdOp->createAndConnectOutTensor(AdamUpdaterOp::getUpdaterOutIndex(),
                                       adamUpdaterId);
  adamUpdOp->setup();
  if (combo->withGradAccum) {
    adamUpdOp->settings.executionContext =
        ExecutionContext::AccumulateOuterFragment;
  }

  // Lamb R1 & R2
  if (combo->mode == AdamMode::Lamb || combo->mode == AdamMode::LambNoBias) {
    auto lambR1OpUp = std::make_unique<LambSquareOp>(
        Op::Settings(graph, combo->name() + "_lamb1"));
    auto lambR1Op = lambR1OpUp.get();
    transferBaseProperties(combo, lambR1Op);
    graph.moveIntoGraph(std::move(lambR1OpUp));

    logging::pattern::trace("Connecting input {} to {} at {}",
                            weightId,
                            lambR1Op->str(),
                            LambSquareOp::getInIndex());
    lambR1Op->connectInTensor(LambSquareOp::getInIndex(), weightId);
    lambR1Op->createAndConnectOutTensor(VarUpdateOp::getUpdatedVarOutIndex(),
                                        lambR1SqId);
    lambR1Op->setup();

    auto lambR2OpUp = std::make_unique<LambSquareOp>(
        Op::Settings(graph, combo->name() + "_lamb2"));
    auto lambR2Op = lambR2OpUp.get();
    transferBaseProperties(combo, lambR2Op);
    graph.moveIntoGraph(std::move(lambR2OpUp));

    logging::pattern::trace("Connecting input {} to {} at {}",
                            adamUpdaterId,
                            lambR2Op->str(),
                            LambSquareOp::getInIndex());
    lambR2Op->connectInTensor(LambSquareOp::getInIndex(), adamUpdaterId);
    lambR2Op->createAndConnectOutTensor(LambSquareOp::getOutIndex(),
                                        lambR2SqId);
    lambR2Op->setup();

    if (combo->withGradAccum) {
      lambR1Op->settings.executionContext =
          ExecutionContext::AccumulateOuterFragment;
      lambR2Op->settings.executionContext =
          ExecutionContext::AccumulateOuterFragment;
    }
  }

  // Var update
  auto adamVarUpdOpUp = std::make_unique<AdamVarUpdateOp>(
      weightId,
      combo->initLr,
      Op::Settings(graph, combo->name() + "_var_update"));
  auto adamVarUpdOp = adamVarUpdOpUp.get();
  transferBaseProperties(combo, adamVarUpdOp);
  graph.moveIntoGraph(std::move(adamVarUpdOpUp));

  // Weight
  logging::pattern::trace("Connecting input {} to {} at {}",
                          weightId,
                          adamUpdOp->str(),
                          AdamVarUpdateOp::getVarToUpdateInIndex());
  adamVarUpdOp->connectInTensor(AdamVarUpdateOp::getVarToUpdateInIndex(),
                                weightId);

  // Updater
  logging::pattern::trace("Connecting input {} to {} at {}",
                          accl1Id,
                          adamUpdOp->str(),
                          AdamVarUpdateOp::getUpdaterInIndex());
  adamVarUpdOp->connectInTensor(AdamVarUpdateOp::getUpdaterInIndex(),
                                adamUpdaterId);

  if (combo->mode == AdamMode::Lamb || combo->mode == AdamMode::LambNoBias) {
    adamVarUpdOp->connectInTensor(AdamVarUpdateOp::getLambR1SqInIndex(),
                                  lambR1SqId);
    adamVarUpdOp->connectInTensor(AdamVarUpdateOp::getLambR2SqInIndex(),
                                  lambR2SqId);
  }

  // Optimizer parameters
  if (!combo->initLr.isConst()) {
    adamVarUpdOp->connectInTensor(AdamVarUpdateOp::getLrInIndex(),
                                  combo->inId(AdamComboOp::getLrInIndex()));
  }

  graph.topoCons->transfer(combo, adamVarUpdOp);

  // deleting combo op now, so that its output can be re-connected
  combo->disconnectAllInputs();
  combo->disconnectAllOutputs();
  graph.eraseOp(combo->id);

  adamVarUpdOp->connectOutTensor(AdamVarUpdateOp::getUpdatedVarOutIndex(),
                                 updatedWeightId);
  adamVarUpdOp->setup();
  if (combo->withGradAccum) {
    adamVarUpdOp->settings.executionContext =
        ExecutionContext::AccumulateOuterFragment;
  }

  return true;
}

namespace {
// Not registering this pattern, as we want it to run at a special time (after
// matmul serialization)
static AddPatternName<AdamDecompose> registerName("AdamDecompose");
} // namespace

} // namespace popart
