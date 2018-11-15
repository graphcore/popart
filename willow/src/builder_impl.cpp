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

std::string BuilderImpl::convolution(const std::string &arg0,
                                     const std::string &arg1,
                                     const std::vector<int> strides,
                                     const std::vector<int> padding,
                                     const std::vector<int> dilation,
                                     int groups) {
  auto id = getNextId();

  auto *graph = model_.mutable_graph();
  auto *node  = graph->add_node();
  node->set_op_type("Conv");
  node->add_input(arg0);
  node->add_input(arg1);
  node->add_output(id);

  auto *auto_pad_attr = node->add_attribute();
  auto_pad_attr->set_name("auto_pad");
  auto_pad_attr->set_type(onnx::AttributeProto::STRING);
  auto_pad_attr->set_s("NOTSET");

  auto *dilations_attr = node->add_attribute();
  dilations_attr->set_name("dilations");
  for (auto i : dilation) {
    dilations_attr->add_ints(i);
  }

  auto *group_attr = node->add_attribute();
  group_attr->set_name("group");
  group_attr->set_i(groups);

  auto *pads_attr = node->add_attribute();
  pads_attr->set_name("pads");
  for (auto i : padding) {
    pads_attr->add_ints(i);
  }

  auto *strides_attr = node->add_attribute();
  strides_attr->set_name("strides");
  for (auto i : strides) {
    strides_attr->add_ints(i);
  }

  onnx::shape_inference::InferShapes(model_);

  return id;
}

std::string BuilderImpl::convolutionWithBias(const std::string &arg0,
                                             const std::string &arg1,
                                             const std::string &arg2,
                                             const std::vector<int> strides,
                                             const std::vector<int> padding,
                                             const std::vector<int> dilation,
                                             int groups) {
  auto id = getNextId();

  auto *graph = model_.mutable_graph();
  auto *node  = graph->add_node();
  node->set_op_type("Conv");
  node->add_input(arg0);
  node->add_input(arg1);
  node->add_input(arg2);
  node->add_output(id);

  auto *auto_pad_attr = node->add_attribute();
  auto_pad_attr->set_name("auto_pad");
  auto_pad_attr->set_type(onnx::AttributeProto::STRING);
  auto_pad_attr->set_s("NOTSET");

  auto *dilations_attr = node->add_attribute();
  dilations_attr->set_name("dilations");
  for (auto i : dilation) {
    dilations_attr->add_ints(i);
  }

  auto *group_attr = node->add_attribute();
  group_attr->set_name("group");
  group_attr->set_i(groups);

  auto *pads_attr = node->add_attribute();
  pads_attr->set_name("pads");
  for (auto i : padding) {
    pads_attr->add_ints(i);
  }

  auto *strides_attr = node->add_attribute();
  strides_attr->set_name("strides");
  for (auto i : strides) {
    strides_attr->add_ints(i);
  }

  onnx::shape_inference::InferShapes(model_);

  return id;
}

std::string BuilderImpl::gemm(const std::string &arg0,
                              const std::string &arg1,
                              const std::string &arg2,
                              float alpha,
                              float beta,
                              int transA,
                              int transB) {
  auto id = getNextId();

  auto *graph = model_.mutable_graph();
  auto *node  = graph->add_node();
  node->set_op_type("GEMM");
  node->add_input(arg0);
  node->add_input(arg1);
  node->add_input(arg2);
  node->add_output(id);

  auto *alpha_attr = node->add_attribute();
  alpha_attr->set_name("alpha");
  alpha_attr->set_f(alpha);

  auto *beta_attr = node->add_attribute();
  beta_attr->set_name("beta");
  beta_attr->set_f(beta);

  auto *transa_attr = node->add_attribute();
  transa_attr->set_name("transA");
  transa_attr->set_i(transA);

  auto *transb_attr = node->add_attribute();
  transb_attr->set_name("transB");
  transb_attr->set_i(transB);

  onnx::shape_inference::InferShapes(model_);

  return id;
}

std::string BuilderImpl::matmul(const std::string &arg0,
                                const std::string &arg1) {
  auto id = getNextId();

  auto *graph = model_.mutable_graph();
  auto *node  = graph->add_node();
  node->set_op_type("MatMul");
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
