#include <poponnx/builder_impl.hpp>
#include <poponnx/error.hpp>
#include <poponnx/tensorinfo.hpp>

#include <onnx/shape_inference/implementation.h>

namespace willow {

static void check_arg_range(const std::vector<std::string> &args,
                            int min,
                            int max,
                            const char *name) {
  auto len = args.size();
  if (len < min || len > max) {
    std::stringstream err;
    err << name << " has invalid number of args. Must be between " << min
        << " and  " << max;
    throw error(err.str());
  }
}

static void check_arg_count(const std::vector<std::string> &args,
                            int count,
                            const char *name) {
  auto len = args.size();
  if (len != count) {
    std::stringstream err;
    err << name << " has invalid number of args. Must be " << count;
    throw error(err.str());
  }
}

static void check_arg_exists(const std::vector<std::string> &args,
                             const char *name) {
  auto len = args.size();
  if (len == 0) {
    std::stringstream err;
    err << name << " has no arguments";
    throw error(err.str());
  }
}

static void add_args(Node *node, const std::vector<std::string> &args) {
  for (const auto &arg : args) {
    node->add_input(arg);
  }
}

std::string BuilderImpl::add_simple_op(const std::vector<std::string> &args,
                                       const char *name,
                                       int arg_count) {
  check_arg_count(args, arg_count, name);

  auto id = getNextId();

  auto *graph = model_.mutable_graph();
  auto *node  = graph->add_node();
  node->set_op_type(name);
  add_args(node, args);
  node->add_output(id);

  onnx::shape_inference::InferShapes(model_);

  return id;
}

std::string BuilderImpl::add_variadic_op(const std::vector<std::string> &args,
                                         const char *name) {
  check_arg_exists(args, name);

  auto id = getNextId();

  auto *graph = model_.mutable_graph();
  auto *node  = graph->add_node();
  node->set_op_type(name);
  add_args(node, args);
  node->add_output(id);

  onnx::shape_inference::InferShapes(model_);

  return id;
}

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

std::string BuilderImpl::abs(const std::vector<std::string> &args) {
  return add_simple_op(args, "Abs", 1);
}

std::string BuilderImpl::acos(const std::vector<std::string> &args) {
  return add_simple_op(args, "Acos", 1);
}

std::string BuilderImpl::acosh(const std::vector<std::string> &args) {
  return add_simple_op(args, "Acosh", 1);
}

std::string BuilderImpl::add(const std::vector<std::string> &args) {
  return add_simple_op(args, "Add", 2);
}

std::string BuilderImpl::logical_and(const std::vector<std::string> &args) {
  return add_simple_op(args, "And", 2);
}

std::string BuilderImpl::asin(const std::vector<std::string> &args) {
  return add_simple_op(args, "Asin", 1);
}

std::string BuilderImpl::asinh(const std::vector<std::string> &args) {
  return add_simple_op(args, "Asinh", 1);
}

std::string BuilderImpl::atan(const std::vector<std::string> &args) {
  return add_simple_op(args, "Atan", 1);
}

std::string BuilderImpl::atanh(const std::vector<std::string> &args) {
  return add_simple_op(args, "Atanh", 1);
}

std::string BuilderImpl::cast(const std::vector<std::string> &args) {
  return add_simple_op(args, "Cast", 1);
}

std::string BuilderImpl::ceil(const std::vector<std::string> &args) {
  return add_simple_op(args, "Ceil", 1);
}

std::string BuilderImpl::cos(const std::vector<std::string> &args) {
  return add_simple_op(args, "Cos", 1);
}

std::string BuilderImpl::cosh(const std::vector<std::string> &args) {
  return add_simple_op(args, "Cosh", 1);
}

std::string BuilderImpl::div(const std::vector<std::string> &args) {
  return add_simple_op(args, "Div", 2);
}

std::string BuilderImpl::elu(const std::vector<std::string> &args) {
  return add_simple_op(args, "Elu", 1);
}

std::string BuilderImpl::equal(const std::vector<std::string> &args) {
  return add_simple_op(args, "Equal", 2);
}

std::string BuilderImpl::exp(const std::vector<std::string> &args) {
  return add_simple_op(args, "Exp", 1);
}

std::string BuilderImpl::floor(const std::vector<std::string> &args) {
  return add_simple_op(args, "Floor", 1);
}

std::string BuilderImpl::greater(const std::vector<std::string> &args) {
  return add_simple_op(args, "Greater", 2);
}

std::string BuilderImpl::identity(const std::vector<std::string> &args) {
  return add_simple_op(args, "Identity", 1);
}

std::string BuilderImpl::less(const std::vector<std::string> &args) {
  return add_simple_op(args, "Less", 2);
}

std::string BuilderImpl::log(const std::vector<std::string> &args) {
  return add_simple_op(args, "Log", 1);
}

std::string BuilderImpl::max(const std::vector<std::string> &args) {
  return add_simple_op(args, "Max", 2);
}

std::string BuilderImpl::mean(const std::vector<std::string> &args) {
  return add_variadic_op(args, "Mean");
}

std::string BuilderImpl::min(const std::vector<std::string> &args) {
  return add_simple_op(args, "Min", 2);
}

std::string BuilderImpl::mul(const std::vector<std::string> &args) {
  return add_simple_op(args, "Mul", 2);
}

std::string BuilderImpl::neg(const std::vector<std::string> &args) {
  return add_simple_op(args, "Neg", 1);
}

std::string BuilderImpl::logical_not(const std::vector<std::string> &args) {
  return add_simple_op(args, "Not", 1);
}

std::string BuilderImpl::logical_or(const std::vector<std::string> &args) {
  return add_simple_op(args, "Or", 2);
}

std::string BuilderImpl::pow(const std::vector<std::string> &args) {
  return add_simple_op(args, "Pow", 2);
}

std::string BuilderImpl::reciprocal(const std::vector<std::string> &args) {
  return add_simple_op(args, "Reciprocal", 1);
}

std::string BuilderImpl::relu(const std::vector<std::string> &args) {
  return add_simple_op(args, "Relu", 1);
}

std::string BuilderImpl::sigmoid(const std::vector<std::string> &args) {
  return add_simple_op(args, "Sigmoid", 1);
}

std::string BuilderImpl::sin(const std::vector<std::string> &args) {
  return add_simple_op(args, "Sin", 1);
}

std::string BuilderImpl::sinh(const std::vector<std::string> &args) {
  return add_simple_op(args, "Sinh", 1);
}

std::string BuilderImpl::softsign(const std::vector<std::string> &args) {
  return add_simple_op(args, "Softsign", 1);
}

std::string BuilderImpl::sqrt(const std::vector<std::string> &args) {
  return add_simple_op(args, "Sqrt", 1);
}

std::string BuilderImpl::sub(const std::vector<std::string> &args) {
  return add_simple_op(args, "Sub", 2);
}

std::string BuilderImpl::sum(const std::vector<std::string> &args) {
  return add_variadic_op(args, "Sum");
}

std::string BuilderImpl::tan(const std::vector<std::string> &args) {
  return add_simple_op(args, "Tan", 1);
}

std::string BuilderImpl::tanh(const std::vector<std::string> &args) {
  return add_simple_op(args, "Tanh", 1);
}

std::string BuilderImpl::logical_xor(const std::vector<std::string> &args) {
  return add_simple_op(args, "Xor", 2);
}

std::string BuilderImpl::convolution(const std::vector<std::string> &args,
                                     const std::vector<int> strides,
                                     const std::vector<int> padding,
                                     const std::vector<int> dilation,
                                     int groups) {
  check_arg_range(args, 2, 3, "Conv");

  auto id = getNextId();

  auto *graph = model_.mutable_graph();
  auto *node  = graph->add_node();
  node->set_op_type("Conv");
  add_args(node, args);
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

std::string BuilderImpl::gemm(const std::vector<std::string> &args,
                              float alpha,
                              float beta,
                              int transA,
                              int transB) {
  check_arg_count(args, 3, "GEMM");

  auto id = getNextId();

  auto *graph = model_.mutable_graph();
  auto *node  = graph->add_node();
  node->set_op_type("GEMM");
  add_args(node, args);
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

std::string BuilderImpl::matmul(const std::vector<std::string> &args) {
  check_arg_count(args, 2, "MatMul");

  auto id = getNextId();

  auto *graph = model_.mutable_graph();
  auto *node  = graph->add_node();
  node->set_op_type("MatMul");
  node->add_input(args[0]);
  node->add_input(args[1]);
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
