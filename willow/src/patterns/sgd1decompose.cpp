#include <memory>
#include <popart/ces/slicece.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/sgd1acclreduce.hpp>
#include <popart/op/sgd1acclupdate.hpp>
#include <popart/op/sgd1accumulate.hpp>
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

  // T12001
  std::vector<T> d(nelms, 0.0f);

  // The weight data:
  const T *weightVal0;

  // It is possible that a transform has resulted in this weight Tensor not
  // having data, as it is the slice of the original weight data. We can try and
  // back-track to find the data. TODO T12031 move the below logic to tensor
  // class and generalize.
  std::vector<char> sliceOutTemp;
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
      sliceOutTemp = ceSlice.compute();
      weightVal0 =
          static_cast<const T *>(static_cast<void *>(sliceOutTemp.data()));
    }
  } else {
    weightVal0 = static_cast<const T *>(weight.tensorData()->data());
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
  auto acclIntoAccumulatorId = reservedAcclToAccumulatorPrefix() + weightGradId;

  // 2) input to AcclReduce (if reduction across across replicas required)
  auto acclIntoReduceId = reservedAcclToReducePrefix() + weightGradId;

  // 3) input to AcclUpdate and VarUpdate
  auto acclIntoUpdateId = reservedAcclToUpdatePrefix() + weightGradId;

  // 4) The output of the AcclUpdateOp
  auto updatedAcclId = reservedAcclFinalOutPrefix() + weightGradId;

  // Create Accumulator Tensor, a Variable Tensor
  if (weightGrad->info.dataType() != weight->info.dataType()) {
    throw error("Currently, weight and weight gradient should have the same "
                "type in SGD1Decompose, this is outstanding work");
  }

  // initialise accumulation tensor, which is not yet in the Ir Graph.
  if (weightGrad->info.dataType() == DataType::FLOAT) {
    addAcclInTensor<float>(*combo, *weight, *weightGrad, acclIntoAccumulatorId);
  } else if (weightGrad->info.dataType() == DataType::FLOAT16) {
    addAcclInTensor<float16_t>(
        *combo, *weight, *weightGrad, acclIntoAccumulatorId);
  } else {
    throw error("Unsupported type in gradient accumulation transformation, "
                "currently only FLOAT16 and FLOAT are supported");
  }
  logging::pattern::trace("Created Accumulator Tensor in SGD1Decompose: {}",
                          acclIntoAccumulatorId);

  // Accumulate Op
  //
  // Inputs:
  // (1) acclIn (a.k.a. the Velocity Tensor / Gradient Accumulation Tensor)
  // (2) dW (a.k.a. the Mini-Batch Weight Gradient Tensor)
  // (3) dampeningScaleFactor (an input only if not Const)
  //
  // Outputs:
  // (4) an alias of acclIn
  auto acclOpUp = std::make_unique<SGD1AccumulateOp>(
      acclIntoAccumulatorId,
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
                          weightGradId);

  // (3)
  if (!combo->initDpsf1.isConst()) {
    acclOp->connectInTensor(
        // the index at which the dampening scale factor is received,
        SGD1AccumulateOp::getDpsf1InIndex(),
        // the name of the dampeninf scale factor, use combo to find this name
        combo->inId(SGD1ComboOp::getDpsf1InIndex()));
  }
  // (4)
  // if there is no AcclReduce, the output goes directly into the updates.
  acclOp->createAndConnectOutTensor(VarUpdateOp::getUpdatedVarOutIndex(),
                                    combo->withAcclReduce ? acclIntoReduceId
                                                          : acclIntoUpdateId);

  // T12001 better encapsulation
  if (ir.additionalModelProtoTensors.find(acclIntoAccumulatorId) ==
      ir.additionalModelProtoTensors.end()) {
    // If we are not going to stream the accl tensors from the host,
    // don't add them to the set of additional tensors to be saved
    // in the onnx modelproto
    if (!ir.streamingIsDisabledForTensor(acclIntoAccumulatorId)) {
      ir.additionalModelProtoTensors.insert(acclIntoAccumulatorId);
    }
  }

  // T12001 confirm that there are no topo cons here rather
  graph.topoCons->transfer(combo, acclOp);
  acclOp->setup();

  if (combo->withAcclReduce) {
    // AcclReduceOp
    //
    // Inputs:
    // (1) redIn
    //
    // Outputs:
    // (2) alias of input
    auto reduceOpUp = std::make_unique<SGD1AcclReduceOp>(
        acclIntoReduceId, Op::Settings(graph, combo->name() + "_reduce"));
    auto reduceOp = reduceOpUp.get();
    transferBaseProperties(combo, reduceOp);
    graph.moveIntoGraph(std::move(reduceOpUp));

    // (1)
    logging::pattern::trace("Connecting input {} to {} at {}",
                            acclIntoReduceId,
                            reduceOp->str(),
                            VarUpdateOp::getVarToUpdateInIndex());
    reduceOp->connectInTensor(VarUpdateOp::getVarToUpdateInIndex(),
                              acclIntoReduceId);

    // (2)
    reduceOp->createAndConnectOutTensor(VarUpdateOp::getUpdatedVarOutIndex(),
                                        acclIntoUpdateId);

    reduceOp->setup();
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
  ir.removeFromTrainTargetOps(combo);
  graph.eraseOp(combo->id);

  // (4)
  sgd1VarUpdateOp->connectOutTensor(VarUpdateOp::getUpdatedVarOutIndex(),
                                    updatedWeightId);
  sgd1VarUpdateOp->setup();

  for (Op *newTarget : std::vector<Op *>{sgd1VarUpdateOp, acclUpdateOp}) {
    if (!ir.addToTrainTargetOps(newTarget)) {
      throw error("Could not add {} to train target ops", newTarget->id);
    }
  }

  // var update before accl update
  graph.topoCons->insert(sgd1VarUpdateOp, acclUpdateOp);

  return true;
}

namespace {
// Not registering this pattern, as we want it to run at a special time (after
// matmul serialization)
}

} // namespace popart
