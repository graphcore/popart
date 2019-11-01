#include <memory>
#include <popart/ces/slicece.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/sgd1acclupdate.hpp>
#include <popart/op/sgd1accumulate.hpp>
#include <popart/op/sgd1varupdate.hpp>
#include <popart/op/sgd1varupdatecombo.hpp>
#include <popart/op/slice.hpp>
#include <popart/patterns/sgd1decompose.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/topocons.hpp>

namespace popart {

bool SGD1Decompose::matches(Op *op) const {
  return op->isConvertibleTo<SGD1VarUpdateComboOp>();
}

std::vector<const Tensor *> SGD1Decompose::touches(Op *) const { return {}; }

namespace {
template <typename T>
void addAcclInTensor(SGD1VarUpdateComboOp &comboOp,
                     const Tensor &weight,
                     const Tensor &weightGrad,
                     const TensorId &acclInId) {

  auto &graph = comboOp.getGraph();
  auto wgInfo = weightGrad.info;
  auto nelms  = wgInfo.nelms();

  // T12001
  std::vector<T> d(nelms, 0.0f);

  // The weight data:
  const T *weightVal0;

  // It is possible that a transform has resulted in this weight Tensor not
  // having data, as it is the slice of the original weight data. We can try and
  // back-track to find the data. TODO T12031 move the below logic to tensor
  // class and generalize.
  if (!weight.hasTensorData()) {
    constexpr const char *infoString =
        "In addAcclInTensor, trying to get the weight's data to initialize the "
        "accumulation tensor with required weight decay term. ";
    if (!weight.hasProducer()) {
      throw error(std::string(infoString) +
                  "But weight has not data. Moreover weight has no "
                  "producer, so can't work back to find data.");
    } else {
      auto asSlice = dynamic_cast<BaseSliceOp *>(weight.getProducer());
      if (!asSlice) {
        throw error(std::string(infoString) +
                    "But it has no data, and it's producer is not a slice to "
                    "work back through, it is a " +
                    weight.getProducer()->str());
      }
      ConstExprSlice ceSlice(asSlice);
      auto out   = ceSlice.compute();
      weightVal0 = static_cast<const T *>(static_cast<void *>(out.data()));
    }
  } else {
    weightVal0 = static_cast<const T *>(weight.tensorData()->data());
  }

  // We add to the initialized velocity a weight decay term (see the equations)
  for (auto i = 0; i < nelms; ++i) {
    // T12001 investigate why += doesn't work
    // Recall, this scaling factor is (1-dm)*wd*vs
    d[i] = weightVal0[i] * static_cast<T>(comboOp.initWdsf1.val());
  }
  graph.getTensors().addVarInit(acclInId, wgInfo, d.data());
}
} // namespace

bool SGD1Decompose::apply(Op *op) const {

  auto &ir    = op->getIr();
  auto &graph = op->getGraph();

  // matches must have verified the correctness before this call
  auto combo = static_cast<SGD1VarUpdateComboOp *>(op);

  Tensor *weightGrad = combo->inTensor(VarUpdateOp::getUpdaterInIndex());
  Tensor *weight     = combo->inTensor(VarUpdateOp::getVarToUpdateInIndex());
  Tensor *newWeight  = combo->outTensor(VarUpdateOp::getUpdatedVarOutIndex());

  TensorId weightGradId    = weightGrad->id;
  TensorId weightId        = weight->id;
  TensorId updatedWeightId = newWeight->id;

  // The Accumulator Tensor (created in this Pattern) goes through 3 stages;
  //
  // 1) Input to SGD1Accumulate, where it will be updated in-place
  auto acclInId = reservedAccumulationPrefix() + weightGradId;
  // 2) Output of SGD1Accumulate, an alias of the input
  auto acclOutId = reservedAccumulationOutPrefix() + weightGradId;
  // 3
  // ) The output of the SGD1AcclUpdateOp, also an alias of the input
  auto updatedAcclId = reservedAccumulationResetPrefix() + weightGradId;

  // Create Accumulator Tensor, a Variable Tensor
  if (weightGrad->info.dataType() != weight->info.dataType()) {
    throw error("Currently, weight and weight gradient should have the same "
                "type in SGD1Decompose, this is outstanding work");
  }

  // initialise accumulation tensor
  if (weightGrad->info.dataType() == DataType::FLOAT) {
    addAcclInTensor<float>(*combo, *weight, *weightGrad, acclInId);
  } else if (weightGrad->info.dataType() == DataType::FLOAT16) {
    addAcclInTensor<float16_t>(*combo, *weight, *weightGrad, acclInId);
  } else {
    throw error("Unsupported type in gradient accumulation transformation, "
                "currently only FLOAT16 and FLOAT are supported");
  }
  logging::pattern::trace("Created Accumulator Tensor in SGD1Decompose: {}",
                          acclInId);

  // Accumulate Op
  //
  // Inputs:
  // (1) acclIn (a.k.a. the Velocity Tensor / Gradient Accumulation Tensor)
  // (2) dW (a.k.a. the Mini-Batch Weight Gradient Tensor)
  // (3) dampeningScaleFactor (an input only if not Const)
  //
  // Outputs:
  // (4) accOut (an alias of acclIn)
  auto acclOpUp = std::make_unique<SGD1AccumulateOp>(
      acclInId, // The accumulator input gets updated
      combo->initDpsf1,
      Op::Settings(graph, combo->name() + "_accumulate"));
  auto acclOp = acclOpUp.get();
  transferBaseProperties(combo, acclOp);
  graph.moveIntoGraph(std::move(acclOpUp));

  // (1)
  acclOp->connectInTensor(VarUpdateOp::getVarToUpdateInIndex(), acclInId);
  logging::pattern::trace("Connecting input {} to {} at {}",
                          acclInId,
                          acclOp->str(),
                          VarUpdateOp::getVarToUpdateInIndex());
  // (2)
  acclOp->connectInTensor(VarUpdateOp::getUpdaterInIndex(), weightGradId);
  logging::pattern::trace("Connecting input {} to {} at {}",
                          weightGradId,
                          acclOp->str(),
                          VarUpdateOp::getUpdaterInIndex());
  // (3)
  if (!combo->initDpsf1.isConst()) {
    acclOp->connectInTensor(
        // the index at which the dampening scale factor is received,
        SGD1AccumulateOp::getDpsf1InIndex(),
        // the name of the dampeninf scale factor, use combo to find this name
        combo->inId(SGD1VarUpdateComboOp::getDpsf1InIndex()));
  }
  // (4)
  acclOp->createAndConnectOutTensor(VarUpdateOp::getUpdatedVarOutIndex(),
                                    acclOutId);

  // T12001 better encapsulation
  if (ir.additionalModelProtoTensors.find(acclInId) ==
      ir.additionalModelProtoTensors.end()) {
    // If we are not going to stream the accl tensors from the host,
    // don't add them to the set of additional tensors to be saved
    // in the onnx modelproto
    if (!ir.streamingIsDisabledForTensor(acclInId)) {
      ir.additionalModelProtoTensors.insert(acclInId);
    }
  }

  // Move constraints from varUpdate to accumulator
  // T12001 confirm that there are no topo cons here rather
  graph.topoCons->transfer(combo, acclOp);
  acclOp->setup();

  const auto &sessionOptions = graph.getIr().getSessionOptions();
  if (sessionOptions.enableReplicatedGraphs &&
      (!acclOp->initDpsf1.isConst() || acclOp->initDpsf1.val() != 0.0f)) {
    throw error("cannot support replication with non-zero dampening");
  }

  // AcclUpdate Op
  //
  // Inputs
  // (1) acclOut (to be scaled by momentum, etc)
  // (2) W
  // (3) momentum (only if not const)
  // (4) weightDecayScaleFactor (only if not const)
  //
  // Outputs
  // (5) acclReset

  auto updateOpUp = std::make_unique<SGD1AcclUpdateOp>(
      acclOutId,
      combo->initMm1,
      combo->initWdsf1,
      Op::Settings(graph, combo->name() + "_accl_update"));
  auto updateOp = updateOpUp.get();
  transferBaseProperties(combo, updateOp);
  graph.moveIntoGraph(std::move(updateOpUp));

  // (1)
  updateOp->connectInTensor(VarUpdateOp::getVarToUpdateInIndex(), acclOutId);
  // (2)
  updateOp->connectInTensor(VarUpdateOp::getUpdaterInIndex(), weightId);
  // (3)
  if (!combo->initMm1.isConst()) {
    updateOp->connectInTensor(
        SGD1AcclUpdateOp::getMm1InIndex(),
        combo->inId(SGD1VarUpdateComboOp::getMm1InIndex()));
  }
  // (4)
  if (!combo->initWdsf1.isConst()) {
    updateOp->connectInTensor(
        SGD1AcclUpdateOp::getWdsf1InIndex(),
        combo->inId(SGD1VarUpdateComboOp::getWdsf1InIndex()));
  }
  // (5)
  updateOp->createAndConnectOutTensor(VarUpdateOp::getUpdatedVarOutIndex(),
                                      updatedAcclId);
  updateOp->setup();

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
  sgd1VarUpdateOp->connectInTensor(VarUpdateOp::getUpdaterInIndex(), acclOutId);
  // (3)
  if (!combo->initSlr1.isConst()) {
    sgd1VarUpdateOp->connectInTensor(
        SGD1VarUpdateOp::getSlr1InIndex(),
        combo->inId(SGD1VarUpdateComboOp::getSlr1InIndex()));
  }

  // deleting combo op now, so that its output can be re-connected
  combo->disconnectAllInputs();
  combo->disconnectAllOutputs();
  ir.removeFromTrainTargetOps(combo);
  graph.eraseOp(combo->id);

  // (4)
  sgd1VarUpdateOp->connectOutTensor(VarUpdateOp::getUpdatedVarOutIndex(),
                                    updatedWeightId);
  sgd1VarUpdateOp->setup();

  for (Op *newTarget : std::vector<Op *>{sgd1VarUpdateOp, updateOp}) {
    if (!ir.addToTrainTargetOps(newTarget)) {
      throw error("Could not add {} to train target ops", newTarget->id);
    }
  }

  // var update before accl update
  graph.topoCons->insert(sgd1VarUpdateOp, updateOp);

  return true;
}

namespace {
// Not registering this pattern, as we want it to run at a special time (after
// matmul serialization)
}

} // namespace popart
