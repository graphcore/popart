#include <onnx/onnx_pb.h>
#include <poponnx/ir.hpp>
#include <poponnx/makeunique.hpp>
#include <poponnx/op/if.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/tensor.hpp>
#include <poponnx/tensors.hpp>

namespace poponnx {

IfOp::IfOp(const OperatorIdentifier &opid_,
           const std::vector<TensorId> &then_input_ids_,
           const std::vector<TensorId> &else_input_ids_,
           const Op::Settings &settings_)
    : Op(opid_, settings_), then_input_ids(then_input_ids_),
      else_input_ids(else_input_ids_) {}

std::unique_ptr<Op> IfOp::clone() const { return make_unique<IfOp>(*this); }

std::vector<std::unique_ptr<Op>> IfOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  return upops;
}

void IfOp::setup() {
  // Connect all the inputs from branch
  // Inputs 1 to (1 + inputsPerBranch()) are the outputs of then_branch
  appendInputs(then_input_ids, getThenScope());
  // Inputs (1 + inputsPerBranch()) to (1 + 2 * inputsPerBranch()) are the
  // outputs of else_branch
  appendInputs(else_input_ids, getElseScope());

  for (int i = 0; i < inputsPerBranch(); i++) {
    outInfo(i) = inInfo(getThenBranchInIndex(i));
  }
}

void IfOp::appendInputs(const std::vector<TensorId> &input_ids,
                        const Scope &scope) {
  for (auto &input_id : input_ids) {
    auto scoped_id = getIr().getTensors().find(input_id, scope);
    connectInTensor(input->n(), scoped_id);
  }
}

// clang-format off
// ((Total # inputs to IfOp) - (one for boolean tensor)) / (2 for number of branches)
// clang-format on
int IfOp::inputsPerBranch() const { return (input->n() - 1) / 2; }

Scope IfOp::getThenScope() { return getScope() / fmt::format("{}_then", id); }

Scope IfOp::getElseScope() { return getScope() / fmt::format("{}_else", id); }

namespace {

static OpCreator<IfOp> ifOpCreator(
    {Onnx::Operators::If_1},
    [](const OperatorIdentifier &opid_,
       const Op::Settings &settings_,
       const Attributes &attr) -> std::unique_ptr<Op> {
      auto else_branch = attr.getAttribute<Attributes::Graph>("else_branch");
      auto then_branch = attr.getAttribute<Attributes::Graph>("then_branch");

      if (else_branch.output().size() != then_branch.output().size()) {
        throw error("IfOp: else_branch and then_branch have different outputs");
      }

      // Collect all input names
      std::vector<TensorId> then_input_ids;
      for (auto &output : then_branch.output()) {
        then_input_ids.push_back(output.name());
      }
      std::vector<TensorId> else_input_ids;
      for (auto &output : else_branch.output()) {
        else_input_ids.push_back(output.name());
      }

      auto op = make_unique<IfOp>(opid_,
                                  std::move(then_input_ids),
                                  std::move(else_input_ids),
                                  settings_);

      // Create the then and else branchs inplace
      settings_.ir.constructFromOnnxGraph(then_branch, op->getThenScope());
      settings_.ir.constructFromOnnxGraph(else_branch, op->getElseScope());

      return std::move(op);
    },
    true);
} // namespace

} // namespace poponnx
