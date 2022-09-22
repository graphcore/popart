// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <memory>
#include <onnxutil.hpp>
#include <string>
#include <vector>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/accumulate.hpp>
#include <popart/op/collectives/replicatedallreduce.hpp>
#include <popart/op/mul.hpp>
#include <popart/op/sgd1acclupdate.hpp>
#include <popart/op/sgd1combo.hpp>
#include <popart/op/sgd1nesterov.hpp>
#include <popart/op/sgd1varupdate.hpp>
#include <popart/optimizer.hpp>
#include <popart/patterns/sgd1decompose.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/topocons.hpp>

#include "popart/datatype.hpp"
#include "popart/error.hpp"
#include "popart/graphcoreoperators.hpp"
#include "popart/half.hpp"
#include "popart/logging.hpp"
#include "popart/op.hpp"
#include "popart/op/varupdate.hpp"
#include "popart/operators.hpp"
#include "popart/optimizervalue.hpp"
#include "popart/patterns/patterns.hpp"
#include "popart/sessionoptions.hpp"
#include "popart/tensordata.hpp"
#include "popart/tensordebuginfo.hpp"
#include "popart/tensornames.hpp"
#include "popart/tensors.hpp"

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
    tempData   = weight.getDataViaGraphTraversal();
    weightVal0 = reinterpret_cast<const T *>(tempData.data());
  }
  // We add to the initialized velocity a weight decay term (see the equations)
  for (auto i = 0; i < nelms; ++i) {
    // TODO T12001: Investigate why += doesn't work
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

  // 6) Nesterov momentum updated gradient if nesterov momentum is enabled
  auto nesterovAcclGradId = reservedAcclPrefix() + weightId + "_nesterovGrad";
  auto nesterovAcclReduceId =
      reservedAcclToReducePrefix() + weightId + "_nesterovGrad";
  auto nesterovAcclUpdateId =
      reservedAcclToUpdatePrefix() + weightId + "_nesterovGrad";
  auto nesterovGradId = weightId + "_nesterovGrad";

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
      if (combo->nesterov) {
        addStateTensor(graph,
                       nesterovAcclGradId,
                       weight->info.shape(),
                       DataType::FLOAT,
                       VariableSettings());
      }
    } else if (weightGrad->info.dataType() == DataType::FLOAT16) {
      addAcclInTensor<float16_t>(
          *combo, *weight, *weightGrad, acclIntoAccumulatorId);
      if (combo->nesterov) {
        addStateTensor(graph,
                       nesterovAcclGradId,
                       weight->info.shape(),
                       DataType::FLOAT16,
                       VariableSettings());
      }
    } else {
      throw error("Unsupported type in gradient accumulation transformation, "
                  "currently only FLOAT16 and FLOAT are supported");
    }
  }

  // Gradient reduction (mutually exclusive with accl reduction)
  if (combo->reductionType == OptimizerReductionType::GradReduce) {
    // GradReduceOp
    auto reduceOp = graph.createOp<ReplicatedAllReduceOp>(
        Onnx::CustomOperators::ReplicatedAllReduce,
        Op::Settings(
            graph, combo->name() + "_reduce", combo->settings.debugInfoId));
    transferBaseProperties(combo, reduceOp);

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
  auto acclOp = graph.createOp<AccumulateOp>(
      AccumulationType::DampenedAdd,
      combo->initDpsf1,
      Op::Settings(
          graph, combo->name() + "_accumulate", combo->settings.debugInfoId));
  transferBaseProperties(combo, acclOp);

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

  // TODO T12001 better encapsulation
  ir.addAdditionalModelProtoTensor(acclIntoAccumulatorId);

  // TODO T12001 confirm that there are no topo cons here rather
  graph.topoCons->transfer(combo, acclOp);
  acclOp->setup();

  if (combo->nesterov) {
    // Accumulate Op
    //
    // Inputs:
    // (1) acclGrad (Gradient Accumulation Tensor)
    // (2) dW (a.k.a. the Mini-Batch Weight Gradient Tensor)
    //
    // Outputs:
    // (4) an alias of acclIn
    auto acclOp = graph.createOp<AccumulateOp>(
        AccumulationType::DampenedAdd,
        OptimizerValue(1.0f),
        Op::Settings(
            graph, combo->name() + "_accumulate", combo->settings.debugInfoId));
    transferBaseProperties(combo, acclOp);

    // (1)
    acclOp->connectInTensor(VarUpdateOp::getVarToUpdateInIndex(),
                            nesterovAcclGradId);
    // (2)
    acclOp->connectInTensor(VarUpdateWithUpdaterOp::getUpdaterInIndex(),
                            combo->reductionType ==
                                    OptimizerReductionType::GradReduce
                                ? reducedWeightGradId
                                : weightGradId);

    // (3)
    // if there is no AcclReduce, the output goes directly into the updates.
    acclOp->createAndConnectOutTensor(VarUpdateOp::getUpdatedVarOutIndex(),
                                      combo->reductionType ==
                                              OptimizerReductionType::AcclReduce
                                          ? nesterovAcclReduceId
                                          : nesterovAcclUpdateId);

    // TODO T12001 better encapsulation
    ir.addAdditionalModelProtoTensor(nesterovAcclGradId);

    // TODO T12001 confirm that there are no topo cons here rather
    graph.topoCons->transfer(combo, acclOp);
    acclOp->setup();
  }

  // Remaining ops run after the gradient accumulation loop (if enabled)

  // Accl reduction (mutually exclusive with gradient reduction)
  if (combo->reductionType == OptimizerReductionType::AcclReduce) {
    // AcclReduceOp
    //
    // Inputs:
    // (1) redIn
    //
    // Outputs:
    // (2) alias of input
    auto reduceOp = graph.createOp<ReplicatedAllReduceInplaceOp>(
        Onnx::CustomOperators::ReplicatedAllReduceInplace,
        Op::Settings(
            graph, combo->name() + "_reduce", combo->settings.debugInfoId));
    transferBaseProperties(combo, reduceOp);

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

    if (combo->nesterov) {
      // AcclReduceOp
      //
      // Inputs:
      // (1) redIn
      //
      // Outputs:
      // (2) alias of input
      auto reduceOp = graph.createOp<ReplicatedAllReduceInplaceOp>(
          Onnx::CustomOperators::ReplicatedAllReduceInplace,
          Op::Settings(
              graph, combo->name() + "_reduce", combo->settings.debugInfoId));
      transferBaseProperties(combo, reduceOp);

      // (1)
      reduceOp->connectInTensor(ReplicatedAllReduceInplaceOp::getInIndex(),
                                nesterovAcclReduceId);

      // (2)
      reduceOp->createAndConnectOutTensor(
          ReplicatedAllReduceInplaceOp::getOutIndex(), nesterovAcclUpdateId);

      reduceOp->setup();
      if (ir.getSessionOptions().enableGradientAccumulation) {
        reduceOp->settings.executionContext =
            ExecutionContext::AccumulateOuterFragment;
        reduceOp->settings.schedulePriority = 0.0;
      }
    }
  }

  if (combo->nesterov) {
    // Get inverse loss scale tensor(Dpsf1 * Ndsf = 1 / ls)
    TensorId sgdInverselossScale = "";
    if (!combo->initDpsf1.isConst()) {
      auto mulOp =
          graph.createOp<MulOp>(Onnx::AiOnnx::OpSet6::Mul,
                                Op::Settings(graph,
                                             combo->name() + "_nesterov_0",
                                             combo->settings.debugInfoId));
      transferBaseProperties(combo, mulOp);
      mulOp->connectInTensor(MulOp::getArg0InIndex(),
                             combo->inId(SGD1ComboOp::getDpsf1InIndex()));
      mulOp->connectInTensor(MulOp::getArg1InIndex(),
                             combo->inId(SGD1ComboOp::getNdsfInIndex()));
      mulOp->createAndConnectOutTensor(MulOp::getOutIndex(),
                                       weightId + "_sgdInverseLossScale");
      mulOp->setup();
      if (ir.getSessionOptions().enableGradientAccumulation) {
        mulOp->settings.executionContext =
            ExecutionContext::AccumulateOuterFragment;
        mulOp->settings.schedulePriority = 0.0;
      }
      sgdInverselossScale = mulOp->outTensor(MulOp::getOutIndex())->id;
    }

    // SGD1Nesterov Op
    //
    // g_out = ngsf * (1 / ls * g + wd * w) + mm * v
    //
    // Inputs
    // (1) g - Gradient
    // (2) w - Weight
    // (3) v - Velocity
    // (4) 1 / ls - Inverse loss scale
    // (5) wd - Weight decay
    // (6) ngsf - Nesterov gradient scale factor(= vs * rf)
    // (7) mm - Momentum
    //
    // Outputs
    // (8) g_out - Nesterov updated gradient
    auto nesterovOp = graph.createOp<SGD1NesterovOp>(
        Onnx::CustomOperators::SGD1Nesterov,
        combo->initNdsf.val() * combo->initDpsf1.val(),
        combo->initWd.val(),
        combo->initNgsf.val(),
        combo->initMm.val(),
        Op::Settings(
            graph, combo->name() + "_nesterov_1", combo->settings.debugInfoId));
    transferBaseProperties(combo, nesterovOp);

    // (1)
    nesterovOp->connectInTensor(SGD1NesterovOp::getGradInIndex(),
                                nesterovAcclUpdateId);
    // (2)
    nesterovOp->connectInTensor(SGD1NesterovOp::getWeightInIndex(), weightId);

    // (3)
    nesterovOp->connectInTensor(SGD1NesterovOp::getVelocityInIndex(),
                                acclIntoUpdateId);

    // (4)
    if (!combo->initNdsf.isConst()) {
      nesterovOp->connectInTensor(SGD1NesterovOp::getInverseLossScaleInIndex(),
                                  sgdInverselossScale);
    }

    // (5)
    if (!combo->initWd.isConst()) {
      nesterovOp->connectInTensor(SGD1NesterovOp::getWdInIndex(),
                                  combo->inId(SGD1ComboOp::getWdInIndex()));
    }

    // (6)
    if (!combo->initNgsf.isConst()) {
      nesterovOp->connectInTensor(SGD1NesterovOp::getNgsfInIndex(),
                                  combo->inId(SGD1ComboOp::getNgsfInIndex()));
    }

    // (7)
    if (!combo->initMm.isConst()) {
      nesterovOp->connectInTensor(SGD1NesterovOp::getMmInIndex(),
                                  combo->inId(SGD1ComboOp::getMmInIndex()));
    }

    // (8)
    nesterovOp->createAndConnectOutTensor(SGD1NesterovOp::getOutIndex(),
                                          nesterovGradId);

    nesterovOp->setup();

    if (ir.getSessionOptions().enableGradientAccumulation) {
      nesterovOp->settings.executionContext =
          ExecutionContext::AccumulateOuterFragment;
      nesterovOp->settings.schedulePriority = 0.0;
    }

    zeroAccumulator(graph, combo, {nesterovOp}, nesterovAcclGradId);
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

  auto acclUpdateOp = graph.createOp<SGD1AcclUpdateOp>(
      combo->initSmm1,
      combo->initSwd1,
      Op::Settings(
          graph, combo->name() + "_accl_update", combo->settings.debugInfoId));
  transferBaseProperties(combo, acclUpdateOp);

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
  auto sgd1VarUpdateOp = graph.createOp<SGD1VarUpdateOp>(
      combo->initSlr1,
      Op::Settings(
          graph, combo->name() + "_var_update", combo->settings.debugInfoId));
  transferBaseProperties(combo, sgd1VarUpdateOp);

  // (1)
  sgd1VarUpdateOp->connectInTensor(VarUpdateOp::getVarToUpdateInIndex(),
                                   weightId);
  // (2)
  sgd1VarUpdateOp->connectInTensor(VarUpdateWithUpdaterOp::getUpdaterInIndex(),
                                   combo->nesterov ? nesterovGradId
                                                   : acclIntoUpdateId);
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
