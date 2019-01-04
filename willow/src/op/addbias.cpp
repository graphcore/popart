#include <algorithm>
#include <numeric>
#include <poponnx/makeunique.hpp>
#include <poponnx/op/addbias.hpp>
#include <poponnx/op/conv.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {

AddBiasOp::AddBiasOp(ConvOp *op_, Attributes attr)
    : Op(Onnx::CustomOperators::AddBias, op_->pir, "", attr) {}

std::unique_ptr<Op> AddBiasOp::clone() const {
  return make_unique<AddBiasOp>(*this);
}

std::vector<std::unique_ptr<Op>> AddBiasOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(make_unique<AddBiasDataGradOp>(this));

  // "Borrowed" from poplin::convolutionBiasUpdate
  std::vector<int64_t> reduceDims(outRank(getOutIndex()) - 1);
  std::iota(reduceDims.begin() + 1, reduceDims.end(), 2);

  upops.emplace_back(make_unique<AddBiasBiasGradOp>(this, reduceDims));
  return upops;
}

void AddBiasOp::setup() { outInfo(getOutIndex()) = inInfo(getDataInIndex()); }

AddBiasBiasGradOp::AddBiasBiasGradOp(AddBiasOp *op_,
                                     const std::vector<int64_t> &_axes)
    : ReduceSumOp(Onnx::CustomGradOperators::AddBiasBiasGrad,
                  op_->pir,
                  _axes,
                  0) {}

const std::map<int, int> &AddBiasBiasGradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {
      {getOutIndex(), AddBiasOp::getBiasInIndex()}};

  return outInfo;
}

const std::vector<GradInOutMapper> &AddBiasBiasGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {getInIndex(), AddBiasOp::getOutIndex(), GradOpInType::GRADOUT}};

  return inInfo;
}

AddBiasDataGradOp::AddBiasDataGradOp(AddBiasOp *op)
    : IdentityOp({Onnx::CustomGradOperators::AddBiasDataGrad, op->pir, {}}) {}

const std::vector<GradInOutMapper> &AddBiasDataGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {getInIndex(), AddBiasOp::getOutIndex(), GradOpInType::GRADOUT}};

  return inInfo;
}

const std::map<int, int> &AddBiasDataGradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {
      {getOutIndex(), AddBiasOp::getDataInIndex()}};

  return outInfo;
}

namespace {

// The AddBiasOp should not be created from the onnx graph
static OpCreator<AddBiasOp>
    addOpCreator(Onnx::CustomOperators::AddBias, nullptr, false);
static GradOpCreator<AddBiasBiasGradOp>
    addBiasBiasGradOpCreator(Onnx::CustomGradOperators::AddBiasBiasGrad);
static GradOpCreator<AddBiasDataGradOp>
    addBiasDataGradOpCreator(Onnx::CustomGradOperators::AddBiasDataGrad);
} // namespace

} // namespace poponnx
