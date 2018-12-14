#include <algorithm>
#include <iterator>
#include <sstream>

#include <poponnx/builder_impl.hpp>
#include <poponnx/error.hpp>
#include <poponnx/onnxutil.hpp>
#include <poponnx/optypes.hpp>
#include <poponnx/tensordata.hpp>
#include <poponnx/tensorinfo.hpp>

#include <iostream>
#include <onnx/checker.h>
#include <onnx/shape_inference/implementation.h>

namespace poponnx {

// Supported IR version
const static uint64_t irVersion = 3;
// Supported operator set version
const static int64_t operatorSetVersion = 9;

static void check_arg_range(const std::vector<TensorId> &args,
                            int min,
                            int max,
                            const char *name) {
  auto len = args.size();
  if (len < min || len > max) {
    throw error("{}  has invalid number of args. Must be between {} and {}",
                name,
                min,
                max);
  }
}

static void check_arg_count(const std::vector<TensorId> &args,
                            int count,
                            const char *name) {
  auto len = args.size();
  if (len != count) {
    throw error("{} has invalid number of args. Must be {}", name, count);
  }
}

static void check_arg_exists(const std::vector<TensorId> &args,
                             const char *name) {
  auto len = args.size();
  if (len == 0) {
    throw error("{} has no arguments", name);
  }
}

static void add_args(Node *node, const std::vector<TensorId> &args) {
  for (const auto &arg : args) {
    node->add_input(arg);
  }
}

TensorId BuilderImpl::add_simple_op(const std::vector<TensorId> &args,
                                    const char *op_type,
                                    int arg_count,
                                    const std::string &name) {
  check_arg_count(args, arg_count, op_type);

  auto id = getNextId();

  auto *graph = model_.mutable_graph();
  auto *node  = graph->add_node();
  node->set_op_type(op_type);
  add_args(node, args);
  node->add_output(id);

  if (!name.empty())
    node->set_name(name);

  onnx::shape_inference::InferShapes(model_);

  return id;
}

TensorId BuilderImpl::add_variadic_op(const std::vector<TensorId> &args,
                                      const char *op_type,
                                      const std::string &name) {
  check_arg_exists(args, op_type);

  auto id = getNextId();

  auto *graph = model_.mutable_graph();
  auto *node  = graph->add_node();
  node->set_op_type(op_type);
  add_args(node, args);
  node->add_output(id);

  if (!name.empty())
    node->set_name(name);

  onnx::shape_inference::InferShapes(model_);

  return id;
}

TensorId BuilderImpl::getNextId() {
  next_id_++;
  return std::to_string(next_id_);
}

BuilderImpl::BuilderImpl() {}

void BuilderImpl::configure() {
  next_id_ = 0;
  model_.set_ir_version(irVersion);
  auto *opset_import = model_.add_opset_import();
  opset_import->set_version(operatorSetVersion);
  model_.mutable_graph()->set_name("BuilderGraph");
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
  initializer->set_data_type(onnxutil::getTPDataType(initData.info.dataType()));
  initializer->set_name(id);

  for (auto d : initData.info.shape()) {
    initializer->add_dims(d);
  }

  int element_count = static_cast<int>(initData.info.nelms());

  switch (initData.info.dataType()) {
  case DataType::FLOAT: {
    auto src = static_cast<const float *>(initData.data);
    auto dst = initializer->mutable_float_data();
    dst->Resize(element_count, 0.0f);
    memcpy(dst->mutable_data(), src, initData.info.nbytes());
    break;
  }
  case DataType::INT32: {
    auto src = static_cast<const int32_t *>(initData.data);
    auto dst = initializer->mutable_int32_data();
    dst->Resize(element_count, 0);
    memcpy(dst->mutable_data(), src, initData.info.nbytes());
    break;
  }
  case DataType::INT64: {
    auto src = static_cast<const int64_t *>(initData.data);
    auto dst = initializer->mutable_int64_data();
    dst->Resize(2 * element_count, 0);
    memcpy(dst->mutable_data(), src, initData.info.nbytes());
    break;
  }
  case DataType::BOOL: {
    auto src = static_cast<const int32_t *>(initData.data);
    auto dst = initializer->mutable_int32_data();
    dst->Resize(element_count, 0);
    memcpy(dst->mutable_data(), src, initData.info.nbytes());
    break;
  }
  case DataType::FLOAT16: {
    auto src = static_cast<const int32_t *>(initData.data);
    auto dst = initializer->mutable_int32_data();
    dst->Resize((element_count + 1) / 2, 0);
    memcpy(dst->mutable_data(), src, initData.info.nbytes());
    break;
  }
  case DataType::UNDEFINED:
  case DataType::UINT8:
  case DataType::INT8:
  case DataType::UINT16:
  case DataType::INT16:
  case DataType::STRING:
  case DataType::DOUBLE:
  case DataType::UINT32:
  case DataType::UINT64:
  case DataType::COMPLEX64:
  case DataType::COMPLEX128:
  case DataType::BFLOAT16:
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

TensorId BuilderImpl::abs(const std::vector<TensorId> &args,
                          const std::string &name) {
  return add_simple_op(args, "Abs", 1, name);
}

TensorId BuilderImpl::acos(const std::vector<TensorId> &args,
                           const std::string &name) {
  return add_simple_op(args, "Acos", 1, name);
}

TensorId BuilderImpl::reshape(const std::vector<TensorId> &args,
                              const std::string &name) {
  return add_simple_op(args, "Reshape", 2, name);
}

TensorId BuilderImpl::acosh(const std::vector<TensorId> &args,
                            const std::string &name) {
  return add_simple_op(args, "Acosh", 1, name);
}

TensorId BuilderImpl::add(const std::vector<TensorId> &args,
                          const std::string &name) {
  return add_simple_op(args, "Add", 2, name);
}

TensorId BuilderImpl::logical_and(const std::vector<TensorId> &args,
                                  const std::string &name) {
  return add_simple_op(args, "And", 2, name);
}

TensorId BuilderImpl::asin(const std::vector<TensorId> &args,
                           const std::string &name) {
  return add_simple_op(args, "Asin", 1, name);
}

TensorId BuilderImpl::asinh(const std::vector<TensorId> &args,
                            const std::string &name) {
  return add_simple_op(args, "Asinh", 1, name);
}

TensorId BuilderImpl::atan(const std::vector<TensorId> &args,
                           const std::string &name) {
  return add_simple_op(args, "Atan", 1, name);
}

TensorId BuilderImpl::atanh(const std::vector<TensorId> &args,
                            const std::string &name) {
  return add_simple_op(args, "Atanh", 1, name);
}

TensorId BuilderImpl::cast(const std::vector<TensorId> &args,
                           const std::string &name) {
  return add_simple_op(args, "Cast", 1, name);
}

TensorId BuilderImpl::ceil(const std::vector<TensorId> &args,
                           const std::string &name) {
  return add_simple_op(args, "Ceil", 1, name);
}

TensorId BuilderImpl::cos(const std::vector<TensorId> &args,
                          const std::string &name) {
  return add_simple_op(args, "Cos", 1, name);
}

TensorId BuilderImpl::cosh(const std::vector<TensorId> &args,
                           const std::string &name) {
  return add_simple_op(args, "Cosh", 1, name);
}

TensorId BuilderImpl::div(const std::vector<TensorId> &args,
                          const std::string &name) {
  return add_simple_op(args, "Div", 2, name);
}

TensorId BuilderImpl::elu(const std::vector<TensorId> &args,
                          const std::string &name) {
  return add_simple_op(args, "Elu", 1, name);
}

TensorId BuilderImpl::equal(const std::vector<TensorId> &args,
                            const std::string &name) {
  return add_simple_op(args, "Equal", 2, name);
}

TensorId BuilderImpl::exp(const std::vector<TensorId> &args,
                          const std::string &name) {
  return add_simple_op(args, "Exp", 1, name);
}

TensorId BuilderImpl::floor(const std::vector<TensorId> &args,
                            const std::string &name) {
  return add_simple_op(args, "Floor", 1, name);
}

TensorId BuilderImpl::greater(const std::vector<TensorId> &args,
                              const std::string &name) {
  return add_simple_op(args, "Greater", 2, name);
}

TensorId BuilderImpl::identity(const std::vector<TensorId> &args,
                               const std::string &name) {
  return add_simple_op(args, "Identity", 1, name);
}

TensorId BuilderImpl::less(const std::vector<TensorId> &args,
                           const std::string &name) {
  return add_simple_op(args, "Less", 2, name);
}

TensorId BuilderImpl::log(const std::vector<TensorId> &args,
                          const std::string &name) {
  return add_simple_op(args, "Log", 1, name);
}

TensorId BuilderImpl::max(const std::vector<TensorId> &args,
                          const std::string &name) {
  return add_simple_op(args, "Max", 2, name);
}

TensorId BuilderImpl::mean(const std::vector<TensorId> &args,
                           const std::string &name) {
  return add_variadic_op(args, "Mean", name);
}

TensorId BuilderImpl::min(const std::vector<TensorId> &args,
                          const std::string &name) {
  return add_simple_op(args, "Min", 2, name);
}

TensorId BuilderImpl::mul(const std::vector<TensorId> &args,
                          const std::string &name) {
  return add_simple_op(args, "Mul", 2, name);
}

TensorId BuilderImpl::neg(const std::vector<TensorId> &args,
                          const std::string &name) {
  return add_simple_op(args, "Neg", 1, name);
}

TensorId BuilderImpl::logical_not(const std::vector<TensorId> &args,
                                  const std::string &name) {
  return add_simple_op(args, "Not", 1, name);
}

TensorId BuilderImpl::logical_or(const std::vector<TensorId> &args,
                                 const std::string &name) {
  return add_simple_op(args, "Or", 2, name);
}

TensorId BuilderImpl::pow(const std::vector<TensorId> &args,
                          const std::string &name) {
  return add_simple_op(args, "Pow", 2, name);
}

TensorId BuilderImpl::reciprocal(const std::vector<TensorId> &args,
                                 const std::string &name) {
  return add_simple_op(args, "Reciprocal", 1, name);
}

TensorId BuilderImpl::relu(const std::vector<TensorId> &args,
                           const std::string &name) {
  return add_simple_op(args, "Relu", 1, name);
}

TensorId BuilderImpl::sigmoid(const std::vector<TensorId> &args,
                              const std::string &name) {
  return add_simple_op(args, "Sigmoid", 1, name);
}

TensorId BuilderImpl::sin(const std::vector<TensorId> &args,
                          const std::string &name) {
  return add_simple_op(args, "Sin", 1, name);
}

TensorId BuilderImpl::sinh(const std::vector<TensorId> &args,
                           const std::string &name) {
  return add_simple_op(args, "Sinh", 1, name);
}

TensorId BuilderImpl::softsign(const std::vector<TensorId> &args,
                               const std::string &name) {
  return add_simple_op(args, "Softsign", 1, name);
}

TensorId BuilderImpl::sqrt(const std::vector<TensorId> &args,
                           const std::string &name) {
  return add_simple_op(args, "Sqrt", 1, name);
}

TensorId BuilderImpl::sub(const std::vector<TensorId> &args,
                          const std::string &name) {
  return add_simple_op(args, "Sub", 2, name);
}

TensorId BuilderImpl::sum(const std::vector<TensorId> &args,
                          const std::string &name) {
  return add_variadic_op(args, "Sum", name);
}

TensorId BuilderImpl::tan(const std::vector<TensorId> &args,
                          const std::string &name) {
  return add_simple_op(args, "Tan", 1, name);
}

TensorId BuilderImpl::tanh(const std::vector<TensorId> &args,
                           const std::string &name) {
  return add_simple_op(args, "Tanh", 1, name);
}

TensorId BuilderImpl::logical_xor(const std::vector<TensorId> &args,
                                  const std::string &name) {
  return add_simple_op(args, "Xor", 2, name);
}

TensorId BuilderImpl::convolution(const std::vector<TensorId> &args,
                                  const std::vector<int64_t> strides,
                                  const std::vector<int64_t> padding,
                                  const std::vector<int64_t> dilation,
                                  int64_t groups,
                                  bool cacheOperation,
                                  const std::string &name) {
  check_arg_range(args, 2, 3, "Conv");

  auto id = getNextId();

  VerifyWindowParameters(args[0], strides, padding, dilation);

  auto *graph = model_.mutable_graph();
  auto *node  = graph->add_node();
  node->set_op_type("Conv");
  add_args(node, args);
  node->add_output(id);

  if (!name.empty())
    node->set_name(name);

  addNodeAttribute("dilations", dilation, {id});
  addNodeAttribute("group", groups, {id});
  addNodeAttribute("pads", padding, {id});
  addNodeAttribute("strides", strides, {id});
  addNodeAttribute(
      "__cache_operation", static_cast<int64_t>(cacheOperation), {id});

  onnx::shape_inference::InferShapes(model_);

  return id;
}

TensorId BuilderImpl::averagepool(const std::vector<TensorId> &args,
                                  const std::vector<int64_t> kernel_shape,
                                  const std::vector<int64_t> strides,
                                  const std::vector<int64_t> padding,
                                  const std::string &name) {
  check_arg_count(args, 1, "AveragePool");

  auto id = getNextId();

  VerifyWindowParameters(args[0], strides, padding);

  auto *graph = model_.mutable_graph();
  auto *node  = graph->add_node();
  node->set_op_type("AveragePool");
  add_args(node, args);
  node->add_output(id);

  if (!name.empty())
    node->set_name(name);

  addNodeAttribute("count_include_pad", static_cast<int64_t>(0), {id});
  addNodeAttribute("kernel_shape", kernel_shape, {id});
  addNodeAttribute("pads", padding, {id});
  addNodeAttribute("strides", strides, {id});

  onnx::shape_inference::InferShapes(model_);

  return id;
}

TensorId BuilderImpl::maxpool(const std::vector<TensorId> &args,
                              const std::vector<int64_t> kernel_shape,
                              const std::vector<int64_t> strides,
                              const std::vector<int64_t> padding,
                              const std::string &name) {
  check_arg_count(args, 1, "MaxPool");

  auto id = getNextId();

  VerifyWindowParameters(args[0], strides, padding);

  auto *graph = model_.mutable_graph();
  auto *node  = graph->add_node();
  node->set_op_type("MaxPool");
  add_args(node, args);
  node->add_output(id);

  if (!name.empty())
    node->set_name(name);

  addNodeAttribute("storage_order", static_cast<int64_t>(0), {id});
  addNodeAttribute("kernel_shape", kernel_shape, {id});
  addNodeAttribute("pads", padding, {id});
  addNodeAttribute("strides", strides, {id});

  onnx::shape_inference::InferShapes(model_);

  return id;
}

TensorId BuilderImpl::gemm(const std::vector<TensorId> &args,
                           float alpha,
                           float beta,
                           int64_t transA,
                           int64_t transB,
                           const std::string &name) {
  check_arg_count(args, 3, "Gemm");

  auto id = getNextId();

  auto *graph = model_.mutable_graph();
  auto *node  = graph->add_node();
  node->set_op_type("Gemm");
  add_args(node, args);
  node->add_output(id);

  if (!name.empty())
    node->set_name(name);

  addNodeAttribute("alpha", alpha, {id});
  addNodeAttribute("beta", beta, {id});
  addNodeAttribute("transA", transA, {id});
  addNodeAttribute("transB", transB, {id});

  onnx::shape_inference::InferShapes(model_);

  return id;
}

TensorId BuilderImpl::pad(const std::vector<TensorId> &args,
                          std::string mode,
                          const std::vector<int64_t> pads,
                          float value,
                          const std::string &name) {
  check_arg_count(args, 1, "Pad");

  auto id = getNextId();

  auto rank = getTensorShape(args[0]).size();
  if (pads.size() != rank * 2) {
    throw error(
        "Padding vector (length {}) doesn't contain 2 entries per input "
        "dimension {}",
        pads.size(),
        rank);
  }

  auto *graph = model_.mutable_graph();
  auto *node  = graph->add_node();
  node->set_op_type("Pad");
  add_args(node, args);
  node->add_output(id);

  if (!name.empty())
    node->set_name(name);

  addNodeAttribute("mode", mode, {id});
  addNodeAttribute("pads", pads, {id});
  addNodeAttribute("value", value, {id});

  onnx::shape_inference::InferShapes(model_);

  return id;
}

TensorId BuilderImpl::matmul(const std::vector<TensorId> &args,
                             const std::string &name) {
  check_arg_count(args, 2, "MatMul");

  auto id = getNextId();

  auto *graph = model_.mutable_graph();
  auto *node  = graph->add_node();
  node->set_op_type("MatMul");
  node->add_input(args[0]);
  node->add_input(args[1]);
  node->add_output(id);

  if (!name.empty())
    node->set_name(name);

  onnx::shape_inference::InferShapes(model_);

  return id;
}

TensorId BuilderImpl::softmax(const std::vector<TensorId> &args,
                              const std::string &name) {
  check_arg_count(args, 1, "Softmax");

  auto id = getNextId();

  auto *graph = model_.mutable_graph();
  auto *node  = graph->add_node();
  node->set_op_type("Softmax");
  add_args(node, args);
  node->add_output(id);

  if (!name.empty())
    node->set_name(name);

  int64_t axis = 1;
  addNodeAttribute("axis", axis, {id});

  onnx::shape_inference::InferShapes(model_);

  return id;
}

TensorId BuilderImpl::subsample(const std::vector<TensorId> &args,
                                const std::vector<int64_t> &strides,
                                const std::string &name) {

  auto id = getNextId();

  auto *graph = model_.mutable_graph();
  auto *node  = graph->add_node();
  node->set_op_type("Subsample");
  node->set_domain(getPoponnxDomain());
  add_args(node, args);
  node->add_output(id);

  if (!name.empty())
    node->set_name(name);

  addNodeAttribute("strides", strides, {id});

  onnx::shape_inference::InferShapes(model_);

  return id;
}

TensorId BuilderImpl::transpose(const std::vector<TensorId> &args,
                                const std::vector<int64_t> &perm,
                                const std::string &name) {
  check_arg_count(args, 1, "Transpose");
  auto id = getNextId();

  auto *graph = model_.mutable_graph();
  auto *node  = graph->add_node();

  node->set_op_type("Transpose");

  add_args(node, args);
  node->add_output(id);

  if (!name.empty())
    node->set_name(name);

  addNodeAttribute("perm", perm, {id});

  onnx::shape_inference::InferShapes(model_);

  return id;
}

Builder::BatchNormalizationTrainingOutputs
BuilderImpl::batchnormalizationTraining(const TensorId x,
                                        const TensorId scale,
                                        const TensorId b,
                                        const TensorId mean,
                                        const TensorId var,
                                        const float epsilon,
                                        const float momentum,
                                        const int spatial,
                                        const std::string &name) {
  Builder::BatchNormalizationTrainingOutputs outputs;

  // Give each output tensor an id
  outputs.y         = getNextId();
  outputs.mean      = getNextId();
  outputs.var       = getNextId();
  outputs.savedMean = getNextId();
  outputs.savedVar  = getNextId();

  auto *graph = model_.mutable_graph();
  auto *node  = graph->add_node();
  node->set_op_type("BatchNormalization");

  node->add_input(x);
  node->add_input(scale);
  node->add_input(b);
  node->add_input(mean);
  node->add_input(var);

  node->add_output(outputs.y);
  node->add_output(outputs.mean);
  node->add_output(outputs.var);
  node->add_output(outputs.savedMean);
  node->add_output(outputs.savedVar);

  if (!name.empty())
    node->set_name(name);

  addNodeAttribute("epsilon", epsilon, *node);
  addNodeAttribute("momentum", momentum, *node);
  addNodeAttribute("spatial", spatial, *node);

  onnx::shape_inference::InferShapes(model_);

  return outputs;
}

TensorId BuilderImpl::batchnormalizationTesting(const TensorId x,
                                                const TensorId scale,
                                                const TensorId b,
                                                const TensorId mean,
                                                const TensorId var,
                                                const float epsilon,
                                                const float momentum,
                                                const int spatial,
                                                const std::string &name) {

  // Give each output tensor an id
  auto y = getNextId();

  auto *graph = model_.mutable_graph();
  auto *node  = graph->add_node();
  node->set_op_type("BatchNormalization");

  node->add_input(x);
  node->add_input(scale);
  node->add_input(b);
  node->add_input(mean);
  node->add_input(var);

  node->add_output(y);

  if (!name.empty())
    node->set_name(name);

  addNodeAttribute("epsilon", epsilon, *node);
  addNodeAttribute("momentum", momentum, *node);
  addNodeAttribute("spatial", spatial, *node);

  onnx::shape_inference::InferShapes(model_);

  return y;
}

bool BuilderImpl::findNodeProtoByOutputNamesImpl(
    onnx::NodeProto *&out,
    const std::set<TensorId> &nodeOutputNames) {
  onnx::GraphProto *graph = model_.mutable_graph();
  for (onnx::NodeProto &node : *graph->mutable_node()) {
    // Don't check nodes which don't have the same number of outputs.
    if (node.output_size() != nodeOutputNames.size()) {
      continue;
    }

    // Match up all the outputs - note that output names are always unique so we
    // don't need to worry about the order.
    std::set<TensorId> unfoundNodeOutputNames = nodeOutputNames;
    for (const std::string &output : node.output()) {
      if (unfoundNodeOutputNames.count(output)) {
        unfoundNodeOutputNames.erase(output);
      }
    }

    // Return the node if we matched.
    if (unfoundNodeOutputNames.size() == 0) {
      out = &node;
      return true;
    }
  }
  return false;
}

onnx::NodeProto &BuilderImpl::findNodeProtoByOutputNames(
    const std::set<TensorId> &nodeOutputNames) {
  onnx::NodeProto *node = nullptr;
  bool found            = findNodeProtoByOutputNamesImpl(node, nodeOutputNames);
  if (!found) {
    std::ostringstream stream;
    std::copy(nodeOutputNames.begin(),
              nodeOutputNames.end(),
              std::ostream_iterator<TensorId>(stream, " ,"));
    std::string s = stream.str();
    s.erase(s.length() - 2);
    throw error("Could not find a node with outputs " + s + ".");
  }
  return *node;
}

bool BuilderImpl::nodeHasAttributeImpl(onnx::AttributeProto *&out,
                                       onnx::NodeProto &node,
                                       const std::string &attributeName) {
  // Finds an attribute in a node.
  for (onnx::AttributeProto &attribute : *node.mutable_attribute()) {
    if (attribute.name().compare(attributeName) == 0) {
      out = &attribute;
      return true;
    }
  }
  return false;
}

bool BuilderImpl::nodeHasAttribute(const std::string &attributeName,
                                   const std::set<TensorId> &nodeOutputNames) {
  onnx::NodeProto &node      = findNodeProtoByOutputNames(nodeOutputNames);
  onnx::AttributeProto *attr = nullptr; // unused
  return nodeHasAttributeImpl(attr, node, attributeName);
}

onnx::AttributeProto &
BuilderImpl::addNewAttributeToNode(const std::string &attributeName,
                                   const std::set<TensorId> &nodeOutputNames) {
  onnx::NodeProto &node = findNodeProtoByOutputNames(nodeOutputNames);
  return addNewAttributeToNode(attributeName, node);
}

onnx::AttributeProto &
BuilderImpl::addNewAttributeToNode(const std::string &attributeName,
                                   onnx::NodeProto &node) {
  onnx::AttributeProto *attr = nullptr;
  bool hasAttribute          = nodeHasAttributeImpl(attr, node, attributeName);
  if (hasAttribute) {
    throw error("Node already has attribute " + attributeName + ".");
  }
  attr = node.add_attribute();
  attr->set_name(attributeName);
  return *attr;
}

void BuilderImpl::addNodeAttribute(const std::string &attributeName,
                                   const int64_t &attributeValue,
                                   const std::set<TensorId> &nodeOutputNames) {
  onnx::AttributeProto &attr =
      addNewAttributeToNode(attributeName, nodeOutputNames);
  attr.set_type(onnx::AttributeProto::INT);
  attr.set_i(attributeValue);
}

void BuilderImpl::addNodeAttribute(const std::string &attributeName,
                                   const int &attributeValue,
                                   const std::set<TensorId> &nodeOutputNames) {
  onnx::AttributeProto &attr =
      addNewAttributeToNode(attributeName, nodeOutputNames);
  attr.set_type(onnx::AttributeProto::INT);
  attr.set_i(attributeValue);
}

void BuilderImpl::addNodeAttribute(const std::string &attributeName,
                                   const int &attributeValue,
                                   onnx::NodeProto &node) {
  onnx::AttributeProto &attr = addNewAttributeToNode(attributeName, node);
  attr.set_type(onnx::AttributeProto::INT);
  attr.set_i(attributeValue);
}

void BuilderImpl::addNodeAttribute(const std::string &attributeName,
                                   const std::vector<int64_t> &attributeValue,
                                   const std::set<TensorId> &nodeOutputNames) {
  onnx::AttributeProto &attr =
      addNewAttributeToNode(attributeName, nodeOutputNames);
  attr.set_type(onnx::AttributeProto::INTS);
  for (int64_t i : attributeValue) {
    attr.add_ints(i);
  }
}

void BuilderImpl::addNodeAttribute(const std::string &attributeName,
                                   const float &attributeValue,
                                   const std::set<TensorId> &nodeOutputNames) {
  onnx::AttributeProto &attr =
      addNewAttributeToNode(attributeName, nodeOutputNames);
  attr.set_type(onnx::AttributeProto::FLOAT);
  attr.set_f(attributeValue);
}

void BuilderImpl::addNodeAttribute(const std::string &attributeName,
                                   const float &attributeValue,
                                   onnx::NodeProto &node) {
  onnx::AttributeProto &attr = addNewAttributeToNode(attributeName, node);
  attr.set_type(onnx::AttributeProto::FLOAT);
  attr.set_f(attributeValue);
}

void BuilderImpl::addNodeAttribute(const std::string &attributeName,
                                   const std::vector<float> &attributeValue,
                                   const std::set<TensorId> &nodeOutputNames) {
  onnx::AttributeProto &attr =
      addNewAttributeToNode(attributeName, nodeOutputNames);
  attr.set_type(onnx::AttributeProto::FLOATS);
  for (float f : attributeValue) {
    attr.add_floats(f);
  }
}

void BuilderImpl::addNodeAttribute(const std::string &attributeName,
                                   const std::string &attributeValue,
                                   const std::set<TensorId> &nodeOutputNames) {
  onnx::AttributeProto &attr =
      addNewAttributeToNode(attributeName, nodeOutputNames);
  attr.set_type(onnx::AttributeProto::STRING);
  attr.set_s(attributeValue);
}

void BuilderImpl::addNodeAttribute(const std::string &attributeName,
                                   const char *attributeValue,
                                   const std::set<TensorId> &nodeOutputNames) {
  onnx::AttributeProto &attr =
      addNewAttributeToNode(attributeName, nodeOutputNames);
  attr.set_type(onnx::AttributeProto::STRING);
  attr.set_s(attributeValue);
}

void BuilderImpl::addNodeAttribute(
    const std::string &attributeName,
    const std::vector<std::string> &attributeValue,
    const std::set<TensorId> &nodeOutputNames) {
  onnx::AttributeProto &attr =
      addNewAttributeToNode(attributeName, nodeOutputNames);
  attr.set_type(onnx::AttributeProto::STRINGS);
  for (std::string s : attributeValue) {
    attr.add_strings(s);
  }
}

void BuilderImpl::addNodeAttribute(const std::string &attributeName,
                                   const bool attributeValue,
                                   const std::set<TensorId> &nodeOutputNames) {
  onnx::AttributeProto &attr =
      addNewAttributeToNode(attributeName, nodeOutputNames);
  attr.set_type(onnx::AttributeProto::INT);
  attr.set_i(static_cast<int>(attributeValue));
}

onnx::AttributeProto &
BuilderImpl::getNodeAttribute(const std::string &attributeName,
                              const std::set<TensorId> &nodeOutputNames) {
  onnx::NodeProto &node      = findNodeProtoByOutputNames(nodeOutputNames);
  onnx::AttributeProto *attr = nullptr;
  bool hasAttribute          = nodeHasAttributeImpl(attr, node, attributeName);
  if (!hasAttribute) {
    throw error("Node does not have an attribute " + attributeName + ".");
  }
  return *attr;
}

int64_t
BuilderImpl::getInt64NodeAttribute(const std::string &attributeName,
                                   const std::set<TensorId> &nodeOutputNames) {
  onnx::AttributeProto &attr = getNodeAttribute(attributeName, nodeOutputNames);
  if (attr.type() != onnx::AttributeProto::INT) {
    throw error("Attribute " + attributeName + " is not an integer.");
  }
  return attr.i();
}

std::vector<int64_t> BuilderImpl::getInt64VectorNodeAttribute(
    const std::string &attributeName,
    const std::set<TensorId> &nodeOutputNames) {
  std::vector<int64_t> out;
  onnx::AttributeProto &attr = getNodeAttribute(attributeName, nodeOutputNames);
  if (attr.type() != onnx::AttributeProto::INTS) {
    throw error("Attribute " + attributeName + " is not an integer vector.");
  }
  for (int64_t i : attr.ints()) {
    out.push_back(i);
  }
  return out;
}

float BuilderImpl::getFloatNodeAttribute(
    const std::string &attributeName,
    const std::set<TensorId> &nodeOutputNames) {
  onnx::AttributeProto &attr = getNodeAttribute(attributeName, nodeOutputNames);
  if (attr.type() != onnx::AttributeProto::FLOAT) {
    throw error("Attribute " + attributeName + " is not a float.");
  }
  return attr.f();
}

std::vector<float> BuilderImpl::getFloatVectorNodeAttribute(
    const std::string &attributeName,
    const std::set<TensorId> &nodeOutputNames) {
  std::vector<float> out;
  onnx::AttributeProto &attr = getNodeAttribute(attributeName, nodeOutputNames);
  if (attr.type() != onnx::AttributeProto::FLOATS) {
    throw error("Attribute " + attributeName + " is not a float vector.");
  }
  for (float f : attr.floats()) {
    out.push_back(f);
  }
  return out;
}

std::string
BuilderImpl::getStringNodeAttribute(const std::string &attributeName,
                                    const std::set<TensorId> &nodeOutputNames) {
  onnx::AttributeProto &attr = getNodeAttribute(attributeName, nodeOutputNames);
  if (attr.type() != onnx::AttributeProto::STRING) {
    throw error("Attribute " + attributeName + " is not a string.");
  }
  return attr.s();
}

std::vector<std::string> BuilderImpl::getStringVectorNodeAttribute(
    const std::string &attributeName,
    const std::set<TensorId> &nodeOutputNames) {
  std::vector<std::string> out;
  onnx::AttributeProto &attr = getNodeAttribute(attributeName, nodeOutputNames);
  if (attr.type() != onnx::AttributeProto::STRINGS) {
    throw error("Attribute " + attributeName + " is not a string vector.");
  }
  for (std::string s : attr.strings()) {
    out.push_back(s);
  }
  return out;
}

bool BuilderImpl::getBoolNodeAttribute(
    const std::string &attributeName,
    const std::set<TensorId> &nodeOutputNames) {

  onnx::AttributeProto &attr = getNodeAttribute(attributeName, nodeOutputNames);
  if (attr.type() != onnx::AttributeProto::INT) {
    throw error("Attribute " + attributeName + " is not a int.");
  }

  return static_cast<bool>(attr.i());
}

void BuilderImpl::removeNodeAttribute(
    const std::string &attributeName,
    const std::set<TensorId> &nodeOutputNames) {
  onnx::NodeProto &node = findNodeProtoByOutputNames(nodeOutputNames);
  // To delete an attribute we must find the iterator for the attribute that we
  // want to delete.
  auto *attrs      = node.mutable_attribute();
  auto attr_it     = attrs->begin();
  auto attr_it_end = attrs->end();
  for (; attr_it != attr_it_end; attr_it++) {
    auto attr = *attr_it;
    if (attr.name().compare(attributeName) == 0) {
      break;
    }
  }
  if (attr_it != attr_it_end) {
    attrs->erase(attr_it);
  } else {
    throw error("Cannot remove attribute " + attributeName +
                " as it does not exist.");
  }
}

std::vector<std::string> BuilderImpl::getAllNodeAttributeNames(
    const std::set<TensorId> &nodeOutputNames) {
  onnx::NodeProto &node = findNodeProtoByOutputNames(nodeOutputNames);
  std::vector<std::string> out;
  for (auto attr : node.attribute()) {
    out.push_back(attr.name());
  }
  return out;
}

// We need to make sure the name translation is unique between different model
// imports.
inline static void
checkUnique(const std::string &name,
            const std::map<std::string, TensorId> &tensorTranslation) {
  if (tensorTranslation.count(name)) {
    throw error("Tensor translation not unique. The name {}  already appeared "
                "in a previously imported model.",
                name);
  }
}

inline static const TensorId
getTranslation(const std::string &name,
               const std::map<std::string, TensorId> &tensorTranslation) {
  auto it = tensorTranslation.find(name);
  if (it == tensorTranslation.end()) {
    throw error("Tensor {} has not been translated.", name);
  }
  return it->second;
}

void BuilderImpl::uniquifyNames(onnx::GraphProto &graph) {
  std::map<std::string, TensorId> currentTensorTranslation;
  // First go through all the inputs.
  for (onnx::ValueInfoProto &vip : *graph.mutable_input()) {
    std::string oldName = vip.name();
    checkUnique(oldName, tensorTranslation_);
    auto newId                        = getNextId();
    currentTensorTranslation[oldName] = newId;
    vip.set_name(newId);
  }

  // Go through all the nodes.
  for (onnx::NodeProto &node : *graph.mutable_node()) {
    // Translates all the inputs - NodeProto should be topologically sorted, so
    // we all node inputs have already been defined.
    for (std::string &name : *node.mutable_input()) {
      name = getTranslation(name, currentTensorTranslation);
    }

    // Translate all the outputs
    for (std::string &name : *node.mutable_output()) {
      auto newId                     = getNextId();
      currentTensorTranslation[name] = newId;
      name                           = newId;
    }
  }

  // Go through all the graph outputs.
  for (onnx::ValueInfoProto &vip : *graph.mutable_output()) {
    std::string oldName = vip.name();
    auto newId          = getTranslation(oldName, currentTensorTranslation);
    vip.set_name(newId);
  }

  // Check the model is still valid after translation.
  onnx::checker::check_model(model_);

  // Merge currentTensorTranslation into tensorTranslation_.
  tensorTranslation_.insert(currentTensorTranslation.begin(),
                            currentTensorTranslation.end());
}

void BuilderImpl::loadModelProto(const std::string &modelProtoOrFilename) {
  // TODO T5564 - merge the models rather than override the existing one.
  model_ = onnxutil::getModelProto(modelProtoOrFilename);

  // Check imported model is valid.
  onnx::checker::check_model(model_);

  // Check the IR version.
  if (model_.ir_version() != irVersion) {
    throw error("Expecting ONNX IR version {}, but got {}.",
                irVersion,
                model_.ir_version());
  }

  // Check the opset versions.
  for (auto opset : model_.opset_import()) {
    if (opset.version() != operatorSetVersion) {
      throw error("Expecting ONNX opset version {}, but got {}.",
                  operatorSetVersion,
                  opset.version());
    }
  }

  if (model_.has_graph()) {
    // We need to make sure all the names are and will be unique - translate
    // them into TensorIDs.
    onnx::GraphProto &graph = *model_.mutable_graph();
    uniquifyNames(graph);
  }
}

const std::map<std::string, TensorId>
BuilderImpl::getTensorTranslation() const {
  return tensorTranslation_;
}

std::string BuilderImpl::getModelProto() const {
  std::string output;
  model_.SerializeToString(&output);
  return output;
}

std::vector<TensorId> BuilderImpl::getInputTensorIds() const {
  std::vector<TensorId> inNames;
  for (const auto &input : model_.graph().input()) {
    inNames.push_back(input.name());
  }
  return inNames;
}

std::vector<TensorId> BuilderImpl::getOutputTensorIds() const {
  std::vector<TensorId> outNames;
  for (const auto &output : model_.graph().output()) {
    outNames.push_back(output.name());
  }
  return outNames;
}

std::vector<TensorId> BuilderImpl::getValueTensorIds() const {
  std::vector<TensorId> valueNames;
  for (const auto &value_info : model_.graph().value_info()) {
    valueNames.push_back(value_info.name());
  }
  return valueNames;
}

bool BuilderImpl::isInputTensor(TensorId id) const {
  std::vector<TensorId> inIds = getInputTensorIds();
  return std::find(inIds.begin(), inIds.end(), id) != inIds.end();
}

bool BuilderImpl::isOutputTensor(TensorId id) const {
  std::vector<TensorId> outIds = getOutputTensorIds();
  return std::find(outIds.begin(), outIds.end(), id) != outIds.end();
}

bool BuilderImpl::isValueTensor(TensorId id) const {
  std::vector<TensorId> valueIds = getValueTensorIds();
  return std::find(valueIds.begin(), valueIds.end(), id) != valueIds.end();
}

std::string BuilderImpl::getStrFromTensorIdVec(std::vector<TensorId> v) const {
  const char *const delim = " ";
  std::ostringstream s;
  std::copy(v.begin(), v.end(), std::ostream_iterator<std::string>(s, delim));
  return s.str();
}

int BuilderImpl::getInputTensorIndex(TensorId id) const {
  if (isInputTensor(id)) {
    int index = -1;
    for (int i = 0; i < model_.graph().input_size(); i++) {
      if (model_.graph().input(i).name() == id) {
        index = i;
      }
    }
    if (index == -1) {
      throw error("{} index not found", id);
    }
    return index;
  } else {
    throw error("{} is not an input tensor. Must be {}",
                id,
                getStrFromTensorIdVec(getInputTensorIds()));
  }
}

int BuilderImpl::getOutputTensorIndex(TensorId id) const {
  if (isOutputTensor(id)) {
    int index = -1;
    for (int i = 0; i < model_.graph().output_size(); i++) {
      if (model_.graph().output(i).name() == id) {
        index = i;
      }
    }
    if (index == -1) {
      throw error("{} index not found", id);
    }
    return index;
  } else {
    throw error("{} is not an output tensor. Must be {}",
                id,
                getStrFromTensorIdVec(getOutputTensorIds()));
  }
}

int BuilderImpl::getValueTensorIndex(TensorId id) const {
  if (isValueTensor(id)) {
    int index = -1;
    for (int i = 0; i < model_.graph().value_info_size(); i++) {
      if (model_.graph().value_info(i).name() == id) {
        index = i;
      }
    }
    if (index == -1) {
      throw error("{} index not found", id);
    }
    return index;
  } else {
    throw error("{} is not an value tensor. Must be {}",
                id,
                getStrFromTensorIdVec(getOutputTensorIds()));
  }
}

const onnx::ValueInfoProto &BuilderImpl::getValueInfoProto(TensorId id) const {
  if (isInputTensor(id)) {
    const onnx::ValueInfoProto &t =
        model_.graph().input(getInputTensorIndex(id));
    return t;
  } else if (isOutputTensor(id)) {
    const onnx::ValueInfoProto &t =
        model_.graph().output(getOutputTensorIndex(id));
    return t;
  } else if (isValueTensor(id)) {
    const onnx::ValueInfoProto &t =
        model_.graph().value_info(getValueTensorIndex(id));
    return t;
  } else {
    throw error("{} is not an known tensor. Must be one of {} {} {}",
                id,
                getStrFromTensorIdVec(getInputTensorIds()),
                getStrFromTensorIdVec(getOutputTensorIds()),
                getStrFromTensorIdVec(getValueTensorIds()));
  }
}

std::vector<int64_t> BuilderImpl::getTensorShape(const TensorId id) {
  std::vector<int64_t> shape;

  auto &t = getValueInfoProto(id);
  for (const auto &dim : t.type().tensor_type().shape().dim()) {
    shape.push_back(dim.dim_value());
  }
  return shape;
}

void BuilderImpl::VerifyWindowParameters(TensorId input,
                                         const std::vector<int64_t> strides,
                                         const std::vector<int64_t> padding,
                                         const std::vector<int64_t> dilation) {
  auto num_spatial_dims = getTensorShape(input).size() - 2;
  if (num_spatial_dims < 1) {
    throw error("Input tensor has no spatial dimensions");
  }
  if (strides.size() != num_spatial_dims) {
    throw error(
        "Length of strides vector {} != number of spatial dimensions {}",
        strides.size(),
        num_spatial_dims);
  }
  if (padding.size() != num_spatial_dims * 2) {
    throw error("Padding vector (length {}) does not have 2 values for each "
                "spatial dimension {}",
                strides.size(),
                num_spatial_dims);
  }
  if (dilation.size() != 0 && dilation.size() != num_spatial_dims) {
    throw error(
        "Length of dilations vector {} != number of spatial dimensions {}",
        strides.size(),
        num_spatial_dims);
  }
}

} // namespace poponnx
