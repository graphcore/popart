#include <algorithm>
#include <vector>
#include <poponnx/makeunique.hpp>
#include <poponnx/op/batchnorm.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/tensor.hpp>
#include <poponnx/tensorindex.hpp>

namespace poponnx {

BatchNormOp::BatchNormOp(const OperatorIdentifier &_opid,
                         Ir *_ir,
                         const std::string &name,
                         const Attributes &_attr)
    : Op(_opid, _ir, name, _attr) {}

std::unique_ptr<Op> BatchNormOp::clone() const {
  return make_unique<BatchNormOp>(*this);
}

std::vector<std::unique_ptr<Op>> BatchNormOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(make_unique<BatchNormGradOp>(this));
  return upops;
}

void BatchNormOp::setup() {

  if (inRank(getXInIndex()) < 4) {
    throw error("batch norm requires a rank > 4. x has rank {}",
                inRank(getXInIndex()));
  }

  // Set the attributes
  nAtts.set(epsilon, "epsilon");
  nAtts.set(momentum, "momentum");
  nAtts.set(spatial, "spatial");

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
  }
}

BatchNormGradOp::BatchNormGradOp(BatchNormOp *op_)
    : Op(Onnx::GradOperators::BatchNormalizationGrad, op_->pir), fwdOp(op_) {}

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

  outInfo(getXOutIndex())     = fwdOp->inInfo(BatchNormOp::getXInIndex());
  outInfo(getScaleOutIndex()) = fwdOp->inInfo(BatchNormOp::getScaleInIndex());
  outInfo(getBOutIndex())     = fwdOp->inInfo(BatchNormOp::getBInIndex());
}

namespace {
static OpCreator<BatchNormOp>
    batchNormOpCreator(Onnx::Operators::BatchNormalization);
static GradOpCreator<BatchNormGradOp>
    batchNormGradOpCreator(Onnx::GradOperators::BatchNormalizationGrad);
} // namespace

} // namespace poponnx
