#include <poponnx/builder_impl.hpp>
#include <poponnx/tensorinfo.hpp>

#include <onnx/shape_inference/implementation.h>

namespace willow {

std::string BuilderImpl::getNextId() {
  next_id_++;
  return std::to_string(next_id_);
}

BuilderImpl::BuilderImpl() : next_id_(0) {
  model_.set_ir_version(3);
  auto *opset_import = model_.add_opset_import();
  opset_import->set_version(9);
}

std::string BuilderImpl::addInputTensor(const TensorInfo &tensorInfo) {
  auto id             = getNextId();
  auto onnxTensorType = tensorInfo.getOnnxTypeProto();

  auto *graph = model_.mutable_graph();
  auto *input = graph->add_input();
  input->set_name(id);

  auto *type = input->mutable_type();
  *type      = onnxTensorType;

  return id;
}

void BuilderImpl::addOutputTensor(const std::string &arg0) {
  auto *graph  = model_.mutable_graph();
  auto *output = graph->add_output();

  bool found = false;
  for (const auto &vi : graph->value_info()) {
    if (vi.name() == arg0) {
      *output = vi;
      found   = true;
    }
  }

  if (!found) {
    output->set_name(arg0);
  }
}

std::string BuilderImpl::add(const std::string &arg0, const std::string &arg1) {
  auto id = getNextId();

  auto *graph = model_.mutable_graph();
  auto *node  = graph->add_node();
  node->set_op_type("Add");
  node->add_input(arg0);
  node->add_input(arg1);
  node->add_output(id);

  onnx::shape_inference::InferShapes(model_);

  return id;
}

std::string BuilderImpl::getModelProto() const {
  std::string output;
  model_.SerializeToString(&output);
  return output;
}

} // namespace willow
