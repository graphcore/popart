#include <willow/add.hpp>
#include <willow/tensor.hpp>

namespace willow {

AddOp::AddOp(const onnx::NodeProto &node, Graph *pgraph) : Op(node, pgraph) {}

std::unique_ptr<Op> AddOp::clone() const {
  return std::unique_ptr<Op>(new AddOp(*this));
}

std::vector<std::unique_ptr<Op>> AddOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(std::unique_ptr<Op>(new AddGradOp(this)));
  return upops;
}

void AddOp::setup() {
  output.tensor(0)->info = npOut(input.tensor(0)->info, input.tensor(1)->info);
}

void AddGradOp::setup() {
  output.tensor(0)->info = addOp->input.tensor(0)->info;
  output.tensor(1)->info = addOp->input.tensor(1)->info;
}

AddGradOp::AddGradOp(AddOp *op_)
    : GradOp({"AddGrad", op_->pgraph, {}, getWillowDomain()}), addOp(op_) {}

Op *AddGradOp::getNonGradCreator() const { return addOp; }

std::map<int, int> AddGradOp::createAddGradOutToIn() const {
  // the grad-op output at index 0 corresponds
  // to the non-grad-op's input at index 0
  // ditto 1.
  return {{0, 0}, {1, 1}};
}

const std::map<int, int> &AddGradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = createAddGradOutToIn();
  return outInfo;
}

const std::vector<GradInOutMapper> &AddGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = createAddGradInfo();
  return inInfo;
}

std::vector<GradInOutMapper> AddGradOp::createAddGradInfo() const {
  // input at index 0 : gradient of output of add
  // might need to reduce across certain axes of this
  // if numpy broadcasting happened
  return {{0, 0, GradOpInType::GRADOUT}};
}

} // namespace willow
