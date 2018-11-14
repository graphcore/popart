#include <algorithm>
#include <numeric>

#include <poponnx/addbias.hpp>
#include <poponnx/conv.hpp>
#include <poponnx/tensor.hpp>

namespace willow {

int AddBiasOp::dataInIndex() { return 0; }
int AddBiasOp::biasInIndex() { return 1; }

AddBiasOp::AddBiasOp(ConvOp *op_)
    : Op({"AddBias", op_->pir, {}, getWillowDomain()}) {}

std::unique_ptr<Op> AddBiasOp::clone() const {
  return std::unique_ptr<Op>(new AddBiasOp(*this));
}

std::vector<std::unique_ptr<Op>> AddBiasOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(new AddBiasDataGradOp(this));

  // "Borrowed" from poplin::convolutionBiasUpdate
  std::vector<int64_t> reduceDims(output.tensor(0)->info.rank() - 1);
  std::iota(reduceDims.begin() + 1, reduceDims.end(), 2);

  upops.emplace_back(new AddBiasBiasGradOp(this, reduceDims));
  return upops;
}

void AddBiasOp::setup() {
  output.tensor(0)->info = input.tensor(dataInIndex())->info;
}

AddBiasBiasGradOp::AddBiasBiasGradOp(AddBiasOp *op_,
                                     const std::vector<int64_t> &_axes)
    : ReduceSumOp({"AddBiasBiasGrad", op_->pir, {}, getWillowDomain()},
                  _axes,
                  0) {}

const std::map<int, int> &AddBiasBiasGradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {{0, AddBiasOp::biasInIndex()}};

  return outInfo;
}

const std::vector<GradInOutMapper> &AddBiasBiasGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {0, 0, GradOpInType::GRADOUT}};

  return inInfo;
}

AddBiasDataGradOp::AddBiasDataGradOp(AddBiasOp *op)
    : IdentityOp({"AddBiasDataGrad", op->pir, {}, getWillowDomain()}) {}

const std::vector<GradInOutMapper> &AddBiasDataGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {0, 0, GradOpInType::GRADOUT}};

  return inInfo;
}

const std::map<int, int> &AddBiasDataGradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {{0, AddBiasOp::dataInIndex()}};

  return outInfo;
}

} // namespace willow
