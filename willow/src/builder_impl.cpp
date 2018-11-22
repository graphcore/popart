#include <poponnx/builder_impl.hpp>
#include <poponnx/error.hpp>
#include <poponnx/tensordata.hpp>
#include <poponnx/tensorinfo.hpp>

#include <onnx/shape_inference/implementation.h>

namespace willow {

static void check_arg_range(const std::vector<TensorId> &args,
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

static void check_arg_count(const std::vector<TensorId> &args,
                            int count,
                            const char *name) {
  auto len = args.size();
  if (len != count) {
    std::stringstream err;
    err << name << " has invalid number of args. Must be " << count;
    throw error(err.str());
  }
}

static void check_arg_exists(const std::vector<TensorId> &args,
                             const char *name) {
  auto len = args.size();
  if (len == 0) {
    std::stringstream err;
    err << name << " has no arguments";
    throw error(err.str());
  }
}

static void add_args(Node *node, const std::vector<TensorId> &args) {
  for (const auto &arg : args) {
    node->add_input(arg);
  }
}

TensorId BuilderImpl::add_simple_op(const std::vector<TensorId> &args,
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

TensorId BuilderImpl::add_variadic_op(const std::vector<TensorId> &args,
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

TensorId BuilderImpl::getNextId() {
  next_id_++;
  return std::to_string(next_id_);
}

BuilderImpl::BuilderImpl() : next_id_(0) {
  model_.set_ir_version(3);
  auto *opset_import = model_.add_opset_import();
  opset_import->set_version(9);
}

TensorId BuilderImpl::addInputTensor(const TensorInfo &tensorInfo) {
  auto id             = getNextId();
  auto onnxTensorType = tensorInfo.getOnnxTypeProto();

  auto *graph = model_.mutable_graph();
  auto *input = graph->add_input();
  input->set_name(id);

  auto *type = input->mutable_type();
  *type      = onnxTensorType;

  return id;
}

TensorId BuilderImpl::addInitializedInputTensor(const ConstVoidData &initData) {
  auto id             = getNextId();
  auto onnxTensorType = initData.info.getOnnxTypeProto();

  auto *graph = model_.mutable_graph();
  auto *input = graph->add_input();
  input->set_name(id);

  auto *type = input->mutable_type();
  *type      = onnxTensorType;

  auto *initializer = graph->add_initializer();
  initializer->set_data_type(initData.info.dataType());

  for (auto d : initData.info.shape()) {
    initializer->add_dims(d);
  }

  int element_count = static_cast<int>(initData.info.nelms());

  switch (initData.info.dataType()) {
  case onnx::TensorProto_DataType_FLOAT: {
    auto src = static_cast<const float *>(initData.data);
    auto dst = initializer->mutable_float_data();
    dst->Resize(element_count, 0.0f);
    memcpy(dst->mutable_data(), src, initData.info.nbytes());
    break;
  }
  case onnx::TensorProto_DataType_INT32: {
    auto src = static_cast<const int32_t *>(initData.data);
    auto dst = initializer->mutable_int32_data();
    dst->Resize(element_count, 0);
    memcpy(dst->mutable_data(), src, initData.info.nbytes());
    break;
  }
  case onnx::TensorProto_DataType_BOOL: {
    auto src = static_cast<const int32_t *>(initData.data);
    auto dst = initializer->mutable_int32_data();
    dst->Resize(element_count, 0);
    memcpy(dst->mutable_data(), src, initData.info.nbytes());
    break;
  }
  case onnx::TensorProto_DataType_FLOAT16: {
    auto src = static_cast<const int32_t *>(initData.data);
    auto dst = initializer->mutable_int32_data();
    dst->Resize((element_count + 1) / 2, 0);
    memcpy(dst->mutable_data(), src, initData.info.nbytes());
    break;
  }
  case onnx::TensorProto_DataType_UNDEFINED:
  case onnx::TensorProto_DataType_UINT8:
  case onnx::TensorProto_DataType_INT8:
  case onnx::TensorProto_DataType_UINT16:
  case onnx::TensorProto_DataType_INT16:
  case onnx::TensorProto_DataType_INT64:
  case onnx::TensorProto_DataType_STRING:
  case onnx::TensorProto_DataType_DOUBLE:
  case onnx::TensorProto_DataType_UINT32:
  case onnx::TensorProto_DataType_UINT64:
  case onnx::TensorProto_DataType_COMPLEX64:
  case onnx::TensorProto_DataType_COMPLEX128:
  case onnx::TensorProto_DataType_BFLOAT16:
    throw error("Unsupported data type in initializer");
  }

  return id;
}

void BuilderImpl::addOutputTensor(const TensorId &arg0) {
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

TensorId BuilderImpl::abs(const std::vector<TensorId> &args) {
  return add_simple_op(args, "Abs", 1);
}

TensorId BuilderImpl::acos(const std::vector<TensorId> &args) {
  return add_simple_op(args, "Acos", 1);
}

TensorId BuilderImpl::acosh(const std::vector<TensorId> &args) {
  return add_simple_op(args, "Acosh", 1);
}

TensorId BuilderImpl::add(const std::vector<TensorId> &args) {
  return add_simple_op(args, "Add", 2);
}

TensorId BuilderImpl::logical_and(const std::vector<TensorId> &args) {
  return add_simple_op(args, "And", 2);
}

TensorId BuilderImpl::asin(const std::vector<TensorId> &args) {
  return add_simple_op(args, "Asin", 1);
}

TensorId BuilderImpl::asinh(const std::vector<TensorId> &args) {
  return add_simple_op(args, "Asinh", 1);
}

TensorId BuilderImpl::atan(const std::vector<TensorId> &args) {
  return add_simple_op(args, "Atan", 1);
}

TensorId BuilderImpl::atanh(const std::vector<TensorId> &args) {
  return add_simple_op(args, "Atanh", 1);
}

TensorId BuilderImpl::cast(const std::vector<TensorId> &args) {
  return add_simple_op(args, "Cast", 1);
}

TensorId BuilderImpl::ceil(const std::vector<TensorId> &args) {
  return add_simple_op(args, "Ceil", 1);
}

TensorId BuilderImpl::cos(const std::vector<TensorId> &args) {
  return add_simple_op(args, "Cos", 1);
}

TensorId BuilderImpl::cosh(const std::vector<TensorId> &args) {
  return add_simple_op(args, "Cosh", 1);
}

TensorId BuilderImpl::div(const std::vector<TensorId> &args) {
  return add_simple_op(args, "Div", 2);
}

TensorId BuilderImpl::elu(const std::vector<TensorId> &args) {
  return add_simple_op(args, "Elu", 1);
}

TensorId BuilderImpl::equal(const std::vector<TensorId> &args) {
  return add_simple_op(args, "Equal", 2);
}

TensorId BuilderImpl::exp(const std::vector<TensorId> &args) {
  return add_simple_op(args, "Exp", 1);
}

TensorId BuilderImpl::floor(const std::vector<TensorId> &args) {
  return add_simple_op(args, "Floor", 1);
}

TensorId BuilderImpl::greater(const std::vector<TensorId> &args) {
  return add_simple_op(args, "Greater", 2);
}

TensorId BuilderImpl::identity(const std::vector<TensorId> &args) {
  return add_simple_op(args, "Identity", 1);
}

TensorId BuilderImpl::less(const std::vector<TensorId> &args) {
  return add_simple_op(args, "Less", 2);
}

TensorId BuilderImpl::log(const std::vector<TensorId> &args) {
  return add_simple_op(args, "Log", 1);
}

TensorId BuilderImpl::max(const std::vector<TensorId> &args) {
  return add_simple_op(args, "Max", 2);
}

TensorId BuilderImpl::mean(const std::vector<TensorId> &args) {
  return add_variadic_op(args, "Mean");
}

TensorId BuilderImpl::min(const std::vector<TensorId> &args) {
  return add_simple_op(args, "Min", 2);
}

TensorId BuilderImpl::mul(const std::vector<TensorId> &args) {
  return add_simple_op(args, "Mul", 2);
}

TensorId BuilderImpl::neg(const std::vector<TensorId> &args) {
  return add_simple_op(args, "Neg", 1);
}

TensorId BuilderImpl::logical_not(const std::vector<TensorId> &args) {
  return add_simple_op(args, "Not", 1);
}

TensorId BuilderImpl::logical_or(const std::vector<TensorId> &args) {
  return add_simple_op(args, "Or", 2);
}

TensorId BuilderImpl::pow(const std::vector<TensorId> &args) {
  return add_simple_op(args, "Pow", 2);
}

TensorId BuilderImpl::reciprocal(const std::vector<TensorId> &args) {
  return add_simple_op(args, "Reciprocal", 1);
}

TensorId BuilderImpl::relu(const std::vector<TensorId> &args) {
  return add_simple_op(args, "Relu", 1);
}

TensorId BuilderImpl::sigmoid(const std::vector<TensorId> &args) {
  return add_simple_op(args, "Sigmoid", 1);
}

TensorId BuilderImpl::sin(const std::vector<TensorId> &args) {
  return add_simple_op(args, "Sin", 1);
}

TensorId BuilderImpl::sinh(const std::vector<TensorId> &args) {
  return add_simple_op(args, "Sinh", 1);
}

TensorId BuilderImpl::softsign(const std::vector<TensorId> &args) {
  return add_simple_op(args, "Softsign", 1);
}

TensorId BuilderImpl::sqrt(const std::vector<TensorId> &args) {
  return add_simple_op(args, "Sqrt", 1);
}

TensorId BuilderImpl::sub(const std::vector<TensorId> &args) {
  return add_simple_op(args, "Sub", 2);
}

TensorId BuilderImpl::sum(const std::vector<TensorId> &args) {
  return add_variadic_op(args, "Sum");
}

TensorId BuilderImpl::tan(const std::vector<TensorId> &args) {
  return add_simple_op(args, "Tan", 1);
}

TensorId BuilderImpl::tanh(const std::vector<TensorId> &args) {
  return add_simple_op(args, "Tanh", 1);
}

TensorId BuilderImpl::logical_xor(const std::vector<TensorId> &args) {
  return add_simple_op(args, "Xor", 2);
}

TensorId BuilderImpl::convolution(const std::vector<TensorId> &args,
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

TensorId BuilderImpl::averagepool(const std::vector<TensorId> &args,
                                  const std::vector<int> kernel_shape,
                                  const std::vector<int> strides,
                                  const std::vector<int> padding) {
  check_arg_count(args, 1, "AveragePool");

  auto id = getNextId();

  auto *graph = model_.mutable_graph();
  auto *node  = graph->add_node();
  node->set_op_type("AveragePool");
  add_args(node, args);
  node->add_output(id);

  auto *auto_pad_attr = node->add_attribute();
  auto_pad_attr->set_name("auto_pad");
  auto_pad_attr->set_type(onnx::AttributeProto::STRING);
  auto_pad_attr->set_s("NOTSET");

  auto *count_include_pad_attr = node->add_attribute();
  count_include_pad_attr->set_name("count_include_pad");
  count_include_pad_attr->set_i(0);

  auto *kernel_shape_attr = node->add_attribute();
  kernel_shape_attr->set_name("kernel_shape");
  for (auto i : kernel_shape) {
    kernel_shape_attr->add_ints(i);
  }

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

TensorId BuilderImpl::maxpool(const std::vector<TensorId> &args,
                              const std::vector<int> kernel_shape,
                              const std::vector<int> strides,
                              const std::vector<int> padding) {
  check_arg_count(args, 1, "MaxPool");

  auto id = getNextId();

  auto *graph = model_.mutable_graph();
  auto *node  = graph->add_node();
  node->set_op_type("MaxPool");
  add_args(node, args);
  node->add_output(id);

  auto *auto_pad_attr = node->add_attribute();
  auto_pad_attr->set_name("auto_pad");
  auto_pad_attr->set_type(onnx::AttributeProto::STRING);
  auto_pad_attr->set_s("NOTSET");

  auto *storage_order_pad_attr = node->add_attribute();
  storage_order_pad_attr->set_name("storage_order");
  storage_order_pad_attr->set_i(0);

  auto *kernel_shape_attr = node->add_attribute();
  kernel_shape_attr->set_name("kernel_shape");
  for (auto i : kernel_shape) {
    kernel_shape_attr->add_ints(i);
  }

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

TensorId BuilderImpl::gemm(const std::vector<TensorId> &args,
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

TensorId BuilderImpl::matmul(const std::vector<TensorId> &args) {
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
