#include <poponnx/add.hpp>
#include <poponnx/tensor.hpp>

namespace willow {

int AddOp::arg0Index() { return 0; }
int AddOp::arg1Index() { return 1; }

AddOp::AddOp(const onnx::NodeProto &node, Ir *ir) : Op(node, ir) {}

std::unique_ptr<Op> AddOp::clone() const {
  return std::unique_ptr<Op>(new AddOp(*this));
}

std::vector<std::unique_ptr<Op>> AddOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;

  const auto &shape_a0 = input.tensor(AddOp::arg0Index())->info.shape();
  const auto &shape_a1 = input.tensor(AddOp::arg1Index())->info.shape();
  const auto &shape_o0 = output.tensor(0)->info.shape();

  upops.emplace_back(std::unique_ptr<Op>(
      new AddArg0GradOp(this, npReductionAxis(shape_a0, shape_o0))));
  upops.emplace_back(std::unique_ptr<Op>(
      new AddArg1GradOp(this, npReductionAxis(shape_a1, shape_o0))));
  return upops;
}

void AddOp::setup() {
  output.tensor(0)->info = npOut(input.tensor(0)->info, input.tensor(1)->info);
}

AddArg0GradOp::AddArg0GradOp(AddOp *op_, const std::vector<int64_t> &axes)
    : ReduceSumOp({"AddArg0Grad", op_->pir, {}, getPoponnxDomain()},
                  axes,
                  false),
      forward_op_arg_info(op_->input.tensor(AddOp::arg0Index())->info) {}

const std::map<int, int> &AddArg0GradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {{0, AddOp::arg0Index()}};
  return outInfo;
}

const std::vector<GradInOutMapper> &AddArg0GradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {0, 0, GradOpInType::GRADOUT}};
  return inInfo;
}

void AddArg0GradOp::setup() { output.tensor(0)->info = forward_op_arg_info; }

AddArg1GradOp::AddArg1GradOp(AddOp *op_, const std::vector<int64_t> &axes)
    : ReduceSumOp({"AddArg1Grad", op_->pir, {}, getPoponnxDomain()},
                  axes,
                  false),
      forward_op_arg_info(op_->input.tensor(AddOp::arg1Index())->info) {}

const std::map<int, int> &AddArg1GradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {{0, AddOp::arg1Index()}};
  return outInfo;
}

const std::vector<GradInOutMapper> &AddArg1GradOp::gradInputInfo() const {
  // input at index 0 : gradient of output of add
  // might need to reduce across certain axes of this
  // if numpy broadcasting happened
  static const std::vector<GradInOutMapper> inInfo = {
      {0, 0, GradOpInType::GRADOUT}};
  return inInfo;
}

void AddArg1GradOp::setup() { output.tensor(0)->info = forward_op_arg_info; }

} // namespace willow
