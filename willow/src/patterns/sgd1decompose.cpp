// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <memory>
#include <popart/ces/concatce.hpp>
#include <popart/ces/flattence.hpp>
#include <popart/ces/slicece.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/onnxutil.hpp>
#include <popart/op/accumulate.hpp>
#include <popart/op/collectives/replicatedallreduce.hpp>
#include <popart/op/concat.hpp>
#include <popart/op/sgd1acclupdate.hpp>
#include <popart/op/sgd1combo.hpp>
#include <popart/op/sgd1varupdate.hpp>
#include <popart/op/slice.hpp>
#include <popart/patterns/sgd1decompose.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/topocons.hpp>

namespace popart {

bool SGD1Decompose::matches(Op *op) const {
  return op->isConvertibleTo<SGD1ComboOp>();
}

std::vector<const Tensor *> SGD1Decompose::touches(Op *) const { return {}; }

namespace {
template <typename T>
void addAcclInTensor(SGD1ComboOp &comboOp,
                     const Tensor &weight,
                     const Tensor &weightGrad,
                     const TensorId &acclIntoAccumulatorId) {

  auto &graph = comboOp.getGraph();
  auto wgInfo = weightGrad.info;
  auto nelms  = wgInfo.nelms();

  // A note: we could have chosen to always initialize the velocity Tensor to
  // 0, which would correspond to no weight decay in the first iteration. One
  // serious issue with this would be that SGD0 and SGD1 behave differently.

  // It is possible that a transform has resulted in this weight Tensor not
  // having data, as it is the slice of the original weight data. We can try and
  // back-track to find the data.
  // tempData needs to be outside the if statment as it should have the same
  // lifetime as weightVal0.
  std::vector<T> d(nelms, 0.0f);
  std::vector<char> tempData;
  const T *weightVal0;
  if (weight.hasTensorData()) {
    const void *outTemp = weight.tensorData()->data();
    weightVal0          = static_cast<const T *>(outTemp);
  } else {
    tempData   = weight.getDataViaRecursion();
    weightVal0 = reinterpret_cast<const T *>(tempData.data());
  }
  // We add to the initialized velocity a weight decay term (see the equations)
  for (auto i = 0; i < nelms; ++i) {
    // T12001 investigate why += doesn't work
    // Recall, this scaling factor is (1-dm)*wd*vs
    d[i] = weightVal0[i] * static_cast<T>(comboOp.initSwd1.val());
  }

  graph.getTensors().addVarInit(acclIntoAccumulatorId, wgInfo, d.data());
}
} // namespace

bool SGD1Decompose::apply(Op *op) const {

  auto &ir    = op->getIr();
  auto &graph = op->getGraph();

  // matches must have verified the correctness before this call
  auto combo = static_cast<SGD1ComboOp *>(op);

  Tensor *weightGrad =
      combo->inTensor(VarUpdateWithUpdaterOp::getUpdaterInIndex());
  Tensor *weight    = combo->inTensor(VarUpdateOp::getVarToUpdateInIndex());
  Tensor *newWeight = combo->outTensor(VarUpdateOp::getUpdatedVarOutIndex());

  TensorId weightGradId    = weightGrad->id;
  TensorId weightId        = weight->id;
  TensorId updatedWeightId = newWeight->id;

  // The Accumulator Tensor (created in this Pattern) goes through 3 ops which
  // update it in-place. At each point it has a different name (aliases of same
  // memory)
  //
  // 1) Input to SGD1Accumulate
  auto acclIntoAccumulatorId = reservedAcclPrefix() + weightId;

  // 2) input to AcclReduce (if reduction across across replicas required)
  auto acclIntoReduceId = reservedAcclToReducePrefix() + weightId;

  // 3) input to AcclUpdate and VarUpdate
  auto acclIntoUpdateId = reservedAcclToUpdatePrefix() + weightId;

  // 4) The output of the AcclUpdateOp
  auto updatedAcclId = reservedAcclFinalOutPrefix() + weightId;

  // 5) Reduced gradient if gradient reduction is used
  auto reducedWeightGradId = weightId + "_reducedGrad";

  // Create Accumulator Tensor, a Variable Tensor
  if (weightGrad->info.dataType() != weight->info.dataType()) {
    throw error("Currently, weight and weight gradient should have the same "
                "type in SGD1Decompose, this is outstanding work");
  }

  // Initialise the accumulation tensors in the Ir...
  if (ir.tensorExistsInInitialisers(acclIntoAccumulatorId)) {
    // ... Either by loading from the onnx model used to create the session,
    // if they already exist
    auto tp = onnxutil::getTensorProto(ir.getModel(), acclIntoAccumulatorId);
    graph.getTensors().addVarInit(acclIntoAccumulatorId, &tp);
  } else {
    // ... Or by initializing directly
    if (weightGrad->info.dataType() == DataType::FLOAT) {
      addAcclInTensor<float>(
          *combo, *weight, *weightGrad, acclIntoAccumulatorId);
    } else if (weightGrad->info.dataType() == DataType::FLOAT16) {
      addAcclInTensor<float16_t>(
          *combo, *weight, *weightGrad, acclIntoAccumulatorId);
    } else {
      throw error("Unsupported type in gradient accumulation transformation, "
                  "currently only FLOAT16 and FLOAT are supported");
    }
  }

  // Gradient reduction (mutually exclusive with accl reduction)
  if (combo->reductionType == OptimizerReductionType::GradReduce) {
    // GradReduceOp
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

    reduceOp->createAndConnectOutTensor(ReplicatedAllReduceOp::getOutIndex(),
                                        reducedWeightGradId);

    reduceOp->setup();

    // Will be transferred to the replacement accl update op, which does
    // not exist yet at this point
    graph.topoCons->insert(reduceOp, combo, true);
  }

  // Accumulate Op
  //
  // Inputs:
  // (1) acclIn (a.k.a. the Velocity Tensor / Gradient Accumulation Tensor)
  // (2) dW (a.k.a. the Mini-Batch Weight Gradient Tensor)
  // (3) dampeningScaleFactor (an input only if not Const)
  //
  // Outputs:
  // (4) an alias of acclIn
  auto acclOpUp = std::make_unique<AccumulateOp>(
      acclIntoAccumulatorId,
      AccumulationType::DampenedAdd,
      combo->initDpsf1,
      Op::Settings(graph, combo->name() + "_accumulate"));
  auto acclOp = acclOpUp.get();
  transferBaseProperties(combo, acclOp);
  graph.moveIntoGraph(std::move(acclOpUp));

  // (1)
  logging::pattern::trace("Connecting input {} to {} at {}",
                          acclIntoAccumulatorId,
                          acclOp->str(),
                          VarUpdateOp::getVarToUpdateInIndex());
  acclOp->connectInTensor(VarUpdateOp::getVarToUpdateInIndex(),
                          acclIntoAccumulatorId);
  // (2)
  logging::pattern::trace("Connecting input {} to {} at {}",
                          weightGradId,
                          acclOp->str(),
                          VarUpdateWithUpdaterOp::getUpdaterInIndex());
  acclOp->connectInTensor(VarUpdateWithUpdaterOp::getUpdaterInIndex(),
                          combo->reductionType ==
                                  OptimizerReductionType::GradReduce
                              ? reducedWeightGradId
                              : weightGradId);

  // (3)
  if (!combo->initDpsf1.isConst()) {
    acclOp->connectInTensor(
        // the index at which the dampening scale factor is received,
        AccumulateOp::getFactorInIndex(),
        // the name of the dampeninf scale factor, use combo to find this name
        combo->inId(SGD1ComboOp::getDpsf1InIndex()));
  }
  // (4)
  // if there is no AcclReduce, the output goes directly into the updates.
  acclOp->createAndConnectOutTensor(VarUpdateOp::getUpdatedVarOutIndex(),
                                    combo->reductionType ==
                                            OptimizerReductionType::AcclReduce
                                        ? acclIntoReduceId
                                        : acclIntoUpdateId);

  // T12001 better encapsulation
  ir.addAdditionalModelProtoTensor(acclIntoAccumulatorId);

  // T12001 confirm that there are no topo cons here rather
  graph.topoCons->transfer(combo, acclOp);
  acclOp->setup();

  // Accl reduction (mutually exclusive with gradient reduction)
  if (combo->reductionType == OptimizerReductionType::AcclReduce) {
    // AcclReduceOp
    //
    // Inputs:
    // (1) redIn
    //
    // Outputs:
    // (2) alias of input
    auto reduceOpUp = std::make_unique<ReplicatedAllReduceInplaceOp>(
        Onnx::CustomOperators::ReplicatedAllReduceInplace,
        Op::Settings(graph, combo->name() + "_reduce"));
    auto reduceOp = reduceOpUp.get();
    transferBaseProperties(combo, reduceOp);
    graph.moveIntoGraph(std::move(reduceOpUp));

    // (1)
    logging::pattern::trace("Connecting input {} to {} at {}",
                            acclIntoReduceId,
                            reduceOp->str(),
                            ReplicatedAllReduceInplaceOp::getInIndex());
    reduceOp->connectInTensor(ReplicatedAllReduceInplaceOp::getInIndex(),
                              acclIntoReduceId);

    // (2)
    reduceOp->createAndConnectOutTensor(
        ReplicatedAllReduceInplaceOp::getOutIndex(), acclIntoUpdateId);

    reduceOp->setup();
    if (ir.getSessionOptions().enableGradientAccumulation) {
      reduceOp->settings.executionContext =
          ExecutionContext::AccumulateOuterFragment;
      reduceOp->settings.schedulePriority = 0.0;
    }
  }

  // AcclUpdate Op
  //
  // Inputs
  // (1) acclIntoUpdateId (to be scaled by momentum, etc)
  // (2) W
  // (3) momentum (only if not const)
  // (4) weightDecayScaleFactor (only if not const)
  //
  // Outputs
  // (5) acclFinal

  auto acclUpdateOpUp = std::make_unique<SGD1AcclUpdateOp>(
      acclIntoUpdateId,
      combo->initSmm1,
      combo->initSwd1,
      Op::Settings(graph, combo->name() + "_accl_update"));
  auto acclUpdateOp = acclUpdateOpUp.get();
  transferBaseProperties(combo, acclUpdateOp);
  graph.moveIntoGraph(std::move(acclUpdateOpUp));

  // (1)
  logging::pattern::trace("Connecting input {} to {} at {}",
                          acclIntoUpdateId,
                          acclUpdateOp->str(),
                          VarUpdateOp::getVarToUpdateInIndex());
  acclUpdateOp->connectInTensor(VarUpdateOp::getVarToUpdateInIndex(),
                                acclIntoUpdateId);
  // (2)
  acclUpdateOp->connectInTensor(VarUpdateWithUpdaterOp::getUpdaterInIndex(),
                                weightId);
  // (3)
  if (!combo->initSmm1.isConst()) {
    acclUpdateOp->connectInTensor(SGD1AcclUpdateOp::getSmm1InIndex(),
                                  combo->inId(SGD1ComboOp::getSmm1InIndex()));
  }
  // (4)
  if (!combo->initSwd1.isConst()) {
    acclUpdateOp->connectInTensor(SGD1AcclUpdateOp::getSwd1InIndex(),
                                  combo->inId(SGD1ComboOp::getSwd1InIndex()));
  }
  // (5)
  acclUpdateOp->createAndConnectOutTensor(VarUpdateOp::getUpdatedVarOutIndex(),
                                          updatedAcclId);
  acclUpdateOp->setup();

  if (ir.getSessionOptions().enableGradientAccumulation) {
    acclUpdateOp->settings.executionContext =
        ExecutionContext::AccumulateOuterFragment;
    acclUpdateOp->settings.schedulePriority = 0.0;
  }

  // VarUpdate
  //
  // Inputs
  // (1) W
  // (2) acclOut
  // (3) scaledLearningRate (an input only if not Const)
  //
  // Outputs
  // (4) W_new
  auto sgd1VarUpdateOpUp = std::make_unique<SGD1VarUpdateOp>(
      weightId,
      combo->initSlr1,
      Op::Settings(graph, combo->name() + "_var_update"));
  auto sgd1VarUpdateOp = sgd1VarUpdateOpUp.get();
  transferBaseProperties(combo, sgd1VarUpdateOp);
  graph.moveIntoGraph(std::move(sgd1VarUpdateOpUp));

  // (1)
  sgd1VarUpdateOp->connectInTensor(VarUpdateOp::getVarToUpdateInIndex(),
                                   weightId);
  // (2)
  sgd1VarUpdateOp->connectInTensor(VarUpdateWithUpdaterOp::getUpdaterInIndex(),
                                   acclIntoUpdateId);
  // (3)
  if (!combo->initSlr1.isConst()) {
    sgd1VarUpdateOp->connectInTensor(
        SGD1VarUpdateOp::getSlr1InIndex(),
        combo->inId(SGD1ComboOp::getSlr1InIndex()));
  }

  // deleting combo op now, so that its output can be re-connected
  combo->disconnectAllInputs();
  combo->disconnectAllOutputs();
  graph.eraseOp(combo->id);

  // (4)
  sgd1VarUpdateOp->connectOutTensor(VarUpdateOp::getUpdatedVarOutIndex(),
                                    updatedWeightId);
  sgd1VarUpdateOp->setup();

  if (ir.getSessionOptions().enableGradientAccumulation) {
    sgd1VarUpdateOp->settings.executionContext =
        ExecutionContext::AccumulateOuterFragment;
    sgd1VarUpdateOp->settings.schedulePriority = 0.0;
  }

  // var update before accl update
  graph.topoCons->insert(sgd1VarUpdateOp, acclUpdateOp);

  return true;
}

namespace {
// Not registering this pattern, as we want it to run at a special time (after
// matmul serialization)
static AddPatternName<SGD1Decompose> registerName("SGD1Decompose");
} // namespace

} // namespace popart
