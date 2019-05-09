#include <algorithm>
#include <vector>
#include <poponnx/graph.hpp>
#include <poponnx/ir.hpp>
#include <poponnx/makeunique.hpp>
#include <poponnx/op/batchnorm.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/opserialiser.hpp>
#include <poponnx/tensor.hpp>
#include <poponnx/tensorindex.hpp>

namespace poponnx {

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
  return make_unique<BatchNormOp>(*this);
}

std::vector<std::unique_ptr<Op>> BatchNormOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(make_unique<BatchNormGradOp>(*this));
  return upops;
}

void BatchNormOp::setup() {

  // The input tensor according to the ONNX spec mush be (N, C, SPATIAL)
  // where SPATIAL is {H, W} for image data
  if (inRank(getXInIndex()) < 3) {
    throw error("batch norm requires X to have a rank >= 3.  x has rank {}",
                inRank(getXInIndex()));
  }

  if (spatial == 0) {
    throw error("batch norm does not currently support spatial set to 0");
  }

  if (spatial == 1) {
    // Add check to make sure that inputs are the right shape
    if (inInfo(getScaleInIndex()).dim(0) != inInfo(getXInIndex()).dim(1)) {
      throw error(
          "batch norm scale dimension 0 ({}) does not equal x dimension 1 ({})",
          inInfo(getScaleInIndex()).dim(0),
          inInfo(getXInIndex()).dim(1));
    }

    if (inInfo(getBInIndex()).dim(0) != inInfo(getXInIndex()).dim(1)) {
      throw error(
          "batch norm b dimension 0 ({}) does not equal x dimension 1 ({})",
          inInfo(getBInIndex()).dim(0),
          inInfo(getXInIndex()).dim(1));
    }

    if (inInfo(getMeanInIndex()).dim(0) != inInfo(getXInIndex()).dim(1)) {
      throw error(
          "batch norm mean dimension 0 ({}) does not equal x dimension 1 ({})",
          inInfo(getMeanInIndex()).dim(0),
          inInfo(getXInIndex()).dim(1));
    }

    if (inInfo(getVarInIndex()).dim(0) != inInfo(getXInIndex()).dim(1)) {
      throw error(
          "batch norm var dimension 0 ({}) does not equal x dimension 1 ({})",
          inInfo(getVarInIndex()).dim(0),
          inInfo(getXInIndex()).dim(1));
    }

  } else {
    // TODO : Add check when spatial is supported
  }

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

    if (inTensor(getMeanInIndex())->tensorType() == TensorType::Variable) {
      auto *variableTensor =
          dynamic_cast<VariableTensor *>(inTensor(getMeanInIndex()));
      variableTensor->setVariableUpdateType(VariableUpdateType::Copy);
      variableTensor->setCopyFromTensor(outId(getMeanOutIndex()));
    }

    if (inTensor(getVarInIndex())->tensorType() == TensorType::Variable) {
      auto *variableTensor =
          dynamic_cast<VariableTensor *>(inTensor(getVarInIndex()));
      variableTensor->setVariableUpdateType(VariableUpdateType::Copy);
      variableTensor->setCopyFromTensor(outId(getVarOutIndex()));
    }
  }

  if (getGraph().getIr().isTraining() && output->n() != 5) {
    throw error(
        "The Ir is in training mode, yet this batch-normalization Op, \"{}\" "
        "has only {} output(s) which means it is in inference mode. To be in "
        "training mode, it should have 5 outputs. Consider using "
        "GraphTransformer::prepareNodesForTraining() to modify the ONNX Model "
        "to have all Nodes set to training mode.",
        str(),
        output->n());
  }
}

void BatchNormOp::appendAttributes(OpSerialiserBase &os) const {
  Op::appendAttributes(os);
  os.appendAttribute("epsilon", epsilon);
  os.appendAttribute("momentum", momentum);
  os.appendAttribute("spatial", spatial);
}

BatchNormGradOp::BatchNormGradOp(const BatchNormOp &op_)
    : Op(Onnx::GradOperators::BatchNormalizationGrad, op_.getSettings()),
      epsilon(op_.getEpsilon()),
      fwdInInfo(op_.inInfo(BatchNormOp::getXInIndex())),
      fwdScaleInInfo(op_.inInfo(BatchNormOp::getScaleInIndex())),
      fwdBInInfo(op_.inInfo(BatchNormOp::getBInIndex())) {}

std::unique_ptr<Op> BatchNormGradOp::clone() const {
  return make_unique<BatchNormGradOp>(*this);
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
      {getYGradInIndex(), BatchNormOp::getYOutIndex(), GradOpInType::GRADOUT},
      {getXInIndex(), BatchNormOp::getXInIndex(), GradOpInType::IN},
      {getScaleInIndex(), BatchNormOp::getScaleInIndex(), GradOpInType::IN},
      {getMeanInIndex(),
       BatchNormOp::getSavedMeanOutIndex(),
       GradOpInType::OUT},
      {getVarInIndex(), BatchNormOp::getSavedVarOutIndex(), GradOpInType::OUT}};

  return inInfo;
}

void BatchNormGradOp::setup() {

  outInfo(getXOutIndex())     = fwdInInfo;
  outInfo(getScaleOutIndex()) = fwdScaleInInfo;
  outInfo(getBOutIndex())     = fwdBInInfo;
}

void BatchNormGradOp::appendAttributes(OpSerialiserBase &os) const {
  Op::appendAttributes(os);
  os.appendAttribute("epsilon", epsilon);
}

namespace {
static OpCreator<BatchNormOp> batchNormOpCreator(
    {Onnx::Operators::BatchNormalization_6,
     Onnx::Operators::BatchNormalization_7,
     Onnx::Operators::BatchNormalization_9},
    [](const OperatorIdentifier &_opid,
       const Op::Settings &settings,
       const Attributes &attr) -> std::unique_ptr<Op> {
      // default epsilon is 10**(-5)
      float epsilon   = attr.getAttribute<Attributes::Float>("epsilon", 1e-5f);
      float momentum  = attr.getAttribute<Attributes::Float>("momentum", 0.9f);
      int64_t spatial = attr.getAttribute<Attributes::Int>("spatial", 1);

      return std::unique_ptr<Op>(
          new BatchNormOp(_opid, epsilon, momentum, spatial, settings));
    },
    true);

} // namespace

} // namespace poponnx
