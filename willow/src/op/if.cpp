#include <onnx/onnx_pb.h>
#include <poponnx/graph.hpp>
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
           const std::vector<TensorId> &then_output_ids_,
           const std::vector<TensorId> &else_output_ids_,
           const Op::Settings &settings_)
    : Op(opid_, settings_), then_input_ids(then_input_ids_),
      else_input_ids(else_input_ids_), then_output_ids(then_output_ids_),
      else_output_ids(else_output_ids_) {}

std::unique_ptr<Op> IfOp::clone() const { return make_unique<IfOp>(*this); }

std::vector<std::unique_ptr<Op>> IfOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  return upops;
}

void IfOp::setup() {
  std::set<TensorId> input_ids;
  input_ids.insert(then_input_ids.begin(), then_input_ids.end());
  input_ids.insert(else_input_ids.begin(), else_input_ids.end());

  for (auto &input_id : input_ids) {
    connectInTensor(input->n(), input_id);
  }

  auto &then_graph = getThenGraph();
  // using the then branch to set output info
  // (guaranteed same output for then and else)
  for (int i = 0; i < then_output_ids.size(); i++) {
    auto &id       = then_output_ids[i];
    auto scoped_id = then_graph.getTensors().find(id, getThenScope());
    auto tensor    = then_graph.getTensors().get(scoped_id);
    outInfo(i)     = tensor->info;
  }
}

Scope IfOp::getThenScope() const {
  return getScope() / fmt::format("{}_then", id);
}

Scope IfOp::getElseScope() const {
  return getScope() / fmt::format("{}_else", id);
}

const Graph &IfOp::getThenGraph() const {
  auto id  = getThenScope().str();
  auto gid = GraphId(id);
  return getGraph().getIr().getGraph(gid);
}

const Graph &IfOp::getElseGraph() const {
  auto id  = getElseScope().str();
  auto gid = GraphId(id);
  return getGraph().getIr().getGraph(gid);
}

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

      auto &tensors = settings_.graph.get().getTensors();
      std::map<TensorId, TensorInfo> input_infos;

      // Collect all input names
      std::vector<TensorId> then_input_ids;
      for (auto &input : then_branch.input()) {
        then_input_ids.push_back(input.name());
        input_infos[input.name()] = tensors.get(input.name())->info;
      }
      std::vector<TensorId> else_input_ids;
      for (auto &input : else_branch.input()) {
        else_input_ids.push_back(input.name());
        input_infos[input.name()] = tensors.get(input.name())->info;
      }

      // Collect all output names
      std::vector<TensorId> then_output_ids;
      for (auto &output : then_branch.output()) {
        then_output_ids.push_back(output.name());
      }
      std::vector<TensorId> else_output_ids;
      for (auto &output : else_branch.output()) {
        else_output_ids.push_back(output.name());
      }

      auto op = make_unique<IfOp>(opid_,
                                  std::move(then_input_ids),
                                  std::move(else_input_ids),
                                  std::move(then_output_ids),
                                  std::move(else_output_ids),
                                  settings_);

      auto &ir         = settings_.graph.get().getIr();
      auto &then_graph = ir.createGraph(GraphId(op->getThenScope().str()));
      for (auto id : then_input_ids) {
        auto scoped_id = (op->getThenScope() / id).str();
        then_graph.addInput(scoped_id, input_infos.at(id));
      }
      then_graph.constructFromOnnxGraph(then_branch, op->getThenScope());

      auto &else_graph = ir.createGraph(GraphId(op->getElseScope().str()));
      for (auto id : else_input_ids) {
        auto scoped_id = (op->getElseScope() / id).str();
        else_graph.addInput(scoped_id, input_infos.at(id));
      }
      else_graph.constructFromOnnxGraph(else_branch, op->getElseScope());

      return std::move(op);
    },
    true);
} // namespace

} // namespace poponnx
