#include <boost/range/numeric.hpp>

#include <memory>
#include <popart/op/split.hpp>
#include <popart/opmanager.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorindex.hpp>

using boost::accumulate;

namespace popart {

SplitOp::SplitOp(const OperatorIdentifier &opid_,
                 int64_t axis_,
                 const std::vector<int64_t> split_,
                 const Op::Settings &settings_)
    : Op(opid_, settings_), axis(axis_), split(split_) {}

void SplitOp::setup() {
  auto numOutputs = output->n();
  auto splitSizes = getSplitSizes();
  if (splitSizes.size() != numOutputs) {
    throw error("Number of outputs does not match number of requested splits");
  }

  auto type  = inInfo(getInIndex()).dataType();
  auto shape = inInfo(getInIndex()).shape();

  if (axis < 0 || axis >= shape.size()) {
    throw error(
        "Axis {} is out of range for tensor with {} dims", axis, shape.size());
  }

  // sum of splitSizes should be equal to axis being split across
  if (accumulate(splitSizes, 0) != shape.at(axis)) {
    throw error("splits {} invalid for dimension of size {}",
                accumulate(splitSizes, 0),
                shape.at(axis));
  }

  for (int i = 0; i < numOutputs; i++) {
    shape[axis] = splitSizes.at(i);
    outInfo(i)  = {type, shape};
  }
}

std::unique_ptr<Op> SplitOp::clone() const {
  return std::make_unique<SplitOp>(*this);
}

std::vector<std::unique_ptr<Op>> SplitOp::getGradOps() {
  std::vector<GradInOutMapper> gradInInfo;
  std::map<int, int> outInfoMap;

  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(std::make_unique<SplitGradOp>(*this, getSettings()));
  return upops;
}

std::vector<int64_t> SplitOp::getSplitSizes() const {
  if (split.size() == 0) {
    auto numOutputs    = output->n();
    auto inShape       = inInfo(getInIndex()).shape();
    auto splitAxisSize = inShape.at(axis);

    if ((splitAxisSize / numOutputs) * numOutputs != splitAxisSize) {
      throw error("Tensor {} ({}) may not be split into equally sized parts "
                  "along axis {}",
                  inTensor(getInIndex())->id,
                  inShape,
                  axis);
    }

    return std::vector<int64_t>(numOutputs, splitAxisSize / numOutputs);
  } else {
    return split;
  }
}

SplitGradOp::SplitGradOp(const SplitOp &fwdOp, const Op::Settings &settings_)
    : Op(Onnx::GradOperators::SplitGrad, settings_),
      fwdOpInInfo(fwdOp.inInfo(SplitOp::getInIndex())), axis(fwdOp.getAxis()) {
  for (int i = 0; i < fwdOp.output->n(); i++) {
    gradInInfo.push_back({i, i, GradOpInType::GRADOUT});
  }

  outInfoMap.insert({getOutIndex(), SplitOp::getInIndex()});
}

void SplitGradOp::setup() { outInfo(getOutIndex()) = fwdOpInInfo; }

std::unique_ptr<Op> SplitGradOp::clone() const {
  return std::make_unique<SplitGradOp>(*this);
}

const std::vector<GradInOutMapper> &SplitGradOp::gradInputInfo() const {
  return gradInInfo;
}

const std::map<int, int> &SplitGradOp::gradOutToNonGradIn() const {
  return outInfoMap;
}

namespace {

static OpCreator<SplitOp> splitOpCreator(
    {Onnx::Operators::Split_2, Onnx::Operators::Split_11},
    [](const OperatorIdentifier &opid_,
       const Op::Settings &settings_,
       const Attributes &attr) -> std::unique_ptr<Op> {
      auto axis  = attr.getAttribute<Attributes::Int>("axis", 0);
      auto split = attr.getAttribute<Attributes::Ints>("split", {});

      return std::make_unique<SplitOp>(opid_, axis, split, settings_);
    },
    true);
} // namespace

} // namespace popart
