// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <memory>
#include <vector>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/batchnorm.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorindex.hpp>

namespace popart {

BatchNormOp::BatchNormOp(const OperatorIdentifier &_opid,
                         float _epsilon,
                         float _momentum,
                         int64_t _spatial,
                         const Op::Settings &settings_)
    : Op(_opid, settings_), epsilon(_epsilon), momentum(_momentum),
      spatial(_spatial) {

  // TODO : T6322 Use the is_training attribute of Version 6
}

std::unique_ptr<Op> BatchNormOp::clone() const {
  return std::make_unique<BatchNormOp>(*this);
}

std::vector<std::unique_ptr<Op>> BatchNormOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(std::make_unique<BatchNormGradOp>(*this));
  return upops;
}

void BatchNormOp::setup() {

  // The input tensor according to the ONNX spec mush be (N, C, SPATIAL)
  // where SPATIAL is {H, W} for image data
  if (inRank(getXInIndex()) < 3) {
    throw error("batch norm requires X to have a rank >= 3.  x has rank {}",
                inRank(getXInIndex()));
  }

  validateInput(inInfo(getScaleInIndex()), "scale");
  validateInput(inInfo(getBInIndex()), "B");
  validateInput(inInfo(getMeanInIndex()), "mean");
  validateInput(inInfo(getVarInIndex()), "var");

  outInfo(getYOutIndex()) = inInfo(getXInIndex());

  if (output->n() > 1) {
    // If we have the optional outputs we can assume we are training
    training                        = true;
    outInfo(getMeanOutIndex())      = inInfo(getMeanInIndex());
    outInfo(getVarOutIndex())       = inInfo(getVarInIndex());
    outInfo(getSavedMeanOutIndex()) = inInfo(getMeanInIndex());
    outInfo(getSavedVarOutIndex())  = inInfo(getVarInIndex());

    // If we have variable input for mean/var then set the update approach to
    // copy
    auto meanTensor = inTensor(getMeanInIndex());
    auto varTensor  = inTensor(getVarInIndex());

    if (meanTensor->tensorType() == TensorType::Variable) {
      meanTensor->setVariableUpdateType(VariableUpdateType::Copy);
      meanTensor->setCopyFromTensor(outId(getMeanOutIndex()));
    }

    if (varTensor->tensorType() == TensorType::Variable) {
      varTensor->setVariableUpdateType(VariableUpdateType::Copy);
      varTensor->setCopyFromTensor(outId(getVarOutIndex()));
    }
  }

  if (getGraph().getIr().isTraining() && output->n() != 5) {
    logging::warn(
        "The Ir is in training mode, yet this batch-normalization Op, \"{}\" "
        "has only {} output(s) which means it is in inference mode. To be in "
        "training mode, it should have 5 outputs. Is this the desired "
        "behaviour? If yes, the op needs to be detached, if not, consider "
        "using GraphTransformer::prepareNodesForTraining() to modify the ONNX "
        "Model to have all Nodes set to training mode.",
        str(),
        output->n());
  }
}

void BatchNormOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("epsilon", epsilon);
  os.appendAttribute("momentum", momentum);
  os.appendAttribute("spatial", spatial);
}

void BatchNormOp::validateInput(const TensorInfo &inputInfo,
                                const std::string &inputName) {

  const TensorInfo &xInfo               = inInfo(getXInIndex());
  const popart::Shape &xShape           = xInfo.shape();
  const popart::Shape &actualInputShape = inputInfo.shape();

  // Work out expected input shape.
  popart::Shape expectedInputShape;
  std::string matchDescription;

  if (getSpatial()) {
    // For spatial=True expect inputs of the shape [C] for x of shape [N, C, D1,
    // ..., Dn].
    expectedInputShape.push_back(xInfo.dim(1));
    matchDescription = "dimension 1";
  } else {
    // For spatial=False expect inputs of the shape [C, D1, ..., Dn] for x of
    // shape [N, C, D1, ..., Dn].
    expectedInputShape.insert(
        expectedInputShape.begin(), xShape.begin() + 1, xShape.end());
    matchDescription = "all but the first dimension";
  }

  if (actualInputShape != expectedInputShape) {
    throw error("expected shape of batch norm input tensor {} ({}) to be {} to "
                "match {} of X ({})",
                inputName,
                actualInputShape,
                expectedInputShape,
                matchDescription,
                xShape);
  }
}

BatchNormGradOp::BatchNormGradOp(const BatchNormOp &op_)
    : Op(Onnx::GradOperators::BatchNormalizationGrad, op_.getSettings()),
      epsilon(op_.getEpsilon()), spatial(op_.getSpatial()),
      fwdInInfo(op_.inInfo(BatchNormOp::getXInIndex())),
      fwdScaleInInfo(op_.inInfo(BatchNormOp::getScaleInIndex())),
      fwdBInInfo(op_.inInfo(BatchNormOp::getBInIndex())) {
  if (op_.output->n() != 5) {
    throw error("BatchNorm Op \"{}\" has {} output(s) which means it is in "
                "inference mode, but its gradient Op is being created. To use "
                "the BatchNorm Op in inference mode during training, it has to "
                "be detached first.",
                op_.str(),
                op_.output->n());
  }
}

std::unique_ptr<Op> BatchNormGradOp::clone() const {
  return std::make_unique<BatchNormGradOp>(*this);
}

const std::map<int, int> &BatchNormGradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {
      {getXOutIndex(), BatchNormOp::getXInIndex()},
      {getScaleOutIndex(), BatchNormOp::getScaleInIndex()},
      {getBOutIndex(), BatchNormOp::getBInIndex()}};
  return outInfo;
}

const std::vector<GradInOutMapper> &BatchNormGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {getYGradInIndex(), BatchNormOp::getYOutIndex(), GradOpInType::GradOut},
      {getXInIndex(), BatchNormOp::getXInIndex(), GradOpInType::In},
      {getScaleInIndex(), BatchNormOp::getScaleInIndex(), GradOpInType::In},
      {getMeanInIndex(),
       BatchNormOp::getSavedMeanOutIndex(),
       GradOpInType::Out},
      {getVarInIndex(), BatchNormOp::getSavedVarOutIndex(), GradOpInType::Out}};

  return inInfo;
}

void BatchNormGradOp::setup() {

  outInfo(getXOutIndex())     = fwdInInfo;
  outInfo(getScaleOutIndex()) = fwdScaleInInfo;
  outInfo(getBOutIndex())     = fwdBInInfo;
}

void BatchNormGradOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("epsilon", epsilon);
}

namespace {

static OpDefinition::DataTypes T = {DataType::FLOAT16, DataType::FLOAT};

static OpDefinition batchNormOpDef(
    {OpDefinition::Inputs({
         {"X", T},
         {"scale", T},
         {"B", T},
         {"mean", T},
         {"var", T},
     }),
     OpDefinition::Outputs({{"Y", T},
                            {"mean", T},
                            {"var", T},
                            {"saved_mean", T},
                            {"saved_var", T}}),
     OpDefinition::Attributes({{"epsilon", {"*"}}, {"momentum", {"*"}}})});

static OpCreator<BatchNormOp> batchNormOpCreator(
    OpDefinitions({
        {Onnx::Operators::BatchNormalization_6, batchNormOpDef},
        {Onnx::Operators::BatchNormalization_7, batchNormOpDef},
        {Onnx::Operators::BatchNormalization_9, batchNormOpDef},
    }),
    [](const OpCreatorInfo &info) {
      // default epsilon is 10**(-5)
      float epsilon =
          info.attributes.getAttribute<Attributes::Float>("epsilon", 1e-5f);
      float momentum =
          info.attributes.getAttribute<Attributes::Float>("momentum", 0.9f);
      int64_t spatial =
          info.attributes.getAttribute<Attributes::Int>("spatial", 1);

      return std::unique_ptr<Op>(new BatchNormOp(
          info.opid, epsilon, momentum, spatial, info.settings));
    },
    true);

} // namespace

} // namespace popart
