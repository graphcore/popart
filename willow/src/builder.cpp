#include <poponnx/builder_impl.hpp>
#include <poponnx/logging.hpp>
#include <poponnx/onnxutil.hpp>
#include <poponnx/opidentifier.hpp>

namespace poponnx {

class TensorInfo;
Builder::Builder() : impl_(new BuilderImpl()) {}

void Builder::configure() { impl_->configure(); }

std::unique_ptr<Builder> Builder::create() {
  auto builder = std::unique_ptr<Builder>(new Builder());
  builder->configure();
  return builder;
}

std::unique_ptr<Builder>
Builder::createFromOnnxModel(const std::string &modelProtoOrFilename) {
  auto builder = create();
  builder->loadModelProto(modelProtoOrFilename);
  return builder;
}

Builder::~Builder() {}

TensorId Builder::addInputTensor(const TensorInfo &tensorInfo) {
  return impl_->addInputTensor(tensorInfo);
}

TensorId Builder::addInitializedInputTensor(const ConstVoidData &initData) {
  return impl_->addInitializedInputTensor(initData);
}

void Builder::addOutputTensor(const TensorId &arg0) {
  return impl_->addOutputTensor(arg0);
}

TensorId Builder::constant(const ConstVoidData &initData,
                           const std::string &name) {
  return impl_->constant(initData, name);
}

TensorId Builder::abs(const std::vector<TensorId> &args,
                      const std::string &name) {
  return impl_->op(Onnx::AiOnnx::OpSet9::Abs, args, {}, name)[0];
}

TensorId Builder::acos(const std::vector<TensorId> &args,
                       const std::string &name) {
  return impl_->op(Onnx::AiOnnx::OpSet9::Acos, args, {}, name)[0];
}

TensorId Builder::acosh(const std::vector<TensorId> &args,
                        const std::string &name) {
  return impl_->op(Onnx::AiOnnx::OpSet9::Acosh, args, {}, name)[0];
}

TensorId Builder::add(const std::vector<TensorId> &args,
                      const std::string &name) {
  return impl_->op(Onnx::AiOnnx::OpSet9::Add, args, {}, name)[0];
}

TensorId Builder::logical_and(const std::vector<TensorId> &args,
                              const std::string &name) {
  return impl_->op(Onnx::AiOnnx::OpSet9::And, args, {}, name)[0];
}

TensorId Builder::asin(const std::vector<TensorId> &args,
                       const std::string &name) {
  return impl_->op(Onnx::AiOnnx::OpSet9::Asin, args, {}, name)[0];
}

TensorId Builder::asinh(const std::vector<TensorId> &args,
                        const std::string &name) {
  return impl_->op(Onnx::AiOnnx::OpSet9::Asinh, args, {}, name)[0];
}

TensorId Builder::atan(const std::vector<TensorId> &args,
                       const std::string &name) {
  return impl_->op(Onnx::AiOnnx::OpSet9::Atan, args, {}, name)[0];
}

TensorId Builder::atanh(const std::vector<TensorId> &args,
                        const std::string &name) {
  return impl_->op(Onnx::AiOnnx::OpSet9::Atanh, args, {}, name)[0];
}

TensorId Builder::cast(const std::vector<TensorId> &args,
                       DataType to,
                       const std::string &name) {
  // getTPDataType returns an enum which needs casting to a int attribute
  return impl_->op(Onnx::AiOnnx::OpSet9::Cast,
                   args,
                   {{"to", static_cast<int>(onnxutil::getTPDataType(to))}},
                   name)[0];
}

TensorId Builder::ceil(const std::vector<TensorId> &args,
                       const std::string &name) {
  return impl_->op(Onnx::AiOnnx::OpSet9::Ceil, args, {}, name)[0];
}

TensorId Builder::concat(const std::vector<TensorId> &args,
                         int64_t dimension,
                         const std::string &name) {
  return impl_->op(
      Onnx::AiOnnx::OpSet9::Concat, args, {{"axis", dimension}}, name)[0];
}

TensorId Builder::cos(const std::vector<TensorId> &args,
                      const std::string &name) {
  return impl_->op(Onnx::AiOnnx::OpSet9::Cos, args, {}, name)[0];
}

TensorId Builder::cosh(const std::vector<TensorId> &args,
                       const std::string &name) {
  return impl_->op(Onnx::AiOnnx::OpSet9::Cosh, args, {}, name)[0];
}

TensorId Builder::div(const std::vector<TensorId> &args,
                      const std::string &name) {
  return impl_->op(Onnx::AiOnnx::OpSet9::Div, args, {}, name)[0];
}

TensorId Builder::elu(const std::vector<TensorId> &args,
                      const std::string &name) {
  return impl_->op(Onnx::AiOnnx::OpSet9::Elu, args, {}, name)[0];
}

TensorId Builder::equal(const std::vector<TensorId> &args,
                        const std::string &name) {
  return impl_->op(Onnx::AiOnnx::OpSet9::Equal, args, {}, name)[0];
}

TensorId Builder::exp(const std::vector<TensorId> &args,
                      const std::string &name) {
  return impl_->op(Onnx::AiOnnx::OpSet9::Exp, args, {}, name)[0];
}

TensorId Builder::floor(const std::vector<TensorId> &args,
                        const std::string &name) {
  return impl_->op(Onnx::AiOnnx::OpSet9::Floor, args, {}, name)[0];
}

TensorId Builder::gather(const std::vector<TensorId> &args,
                         int64_t axis,
                         const std::string &name) {
  return impl_->op(
      Onnx::AiOnnx::OpSet9::Gather, args, {{"axis", axis}}, name)[0];
}

TensorId Builder::greater(const std::vector<TensorId> &args,
                          const std::string &name) {
  return impl_->op(Onnx::AiOnnx::OpSet9::Greater, args, {}, name)[0];
}

TensorId Builder::identity(const std::vector<TensorId> &args,
                           const std::string &name) {
  return impl_->op(Onnx::AiOnnx::OpSet9::Identity, args, {}, name)[0];
}

TensorId Builder::less(const std::vector<TensorId> &args,
                       const std::string &name) {
  return impl_->op(Onnx::AiOnnx::OpSet9::Less, args, {}, name)[0];
}

TensorId Builder::log(const std::vector<TensorId> &args,
                      const std::string &name) {
  return impl_->op(Onnx::AiOnnx::OpSet9::Log, args, {}, name)[0];
}

TensorId Builder::logsoftmax(const std::vector<TensorId> &args,
                             const std::string &name) {
  return impl_->op(
      Onnx::AiOnnx::OpSet9::LogSoftmax, args, {{"axis", 1}}, name)[0];
}

TensorId Builder::max(const std::vector<TensorId> &args,
                      const std::string &name) {
  return impl_->op(Onnx::AiOnnx::OpSet9::Max, args, {}, name)[0];
}

TensorId Builder::mean(const std::vector<TensorId> &args,
                       const std::string &name) {
  return impl_->op(Onnx::AiOnnx::OpSet9::Mean, args, {}, name)[0];
}

TensorId Builder::min(const std::vector<TensorId> &args,
                      const std::string &name) {
  return impl_->op(Onnx::AiOnnx::OpSet9::Min, args, {}, name)[0];
}

TensorId Builder::mul(const std::vector<TensorId> &args,
                      const std::string &name) {
  return impl_->op(Onnx::AiOnnx::OpSet9::Mul, args, {}, name)[0];
}

TensorId Builder::neg(const std::vector<TensorId> &args,
                      const std::string &name) {
  return impl_->op(Onnx::AiOnnx::OpSet9::Neg, args, {}, name)[0];
}

TensorId Builder::logical_not(const std::vector<TensorId> &args,
                              const std::string &name) {
  return impl_->op(Onnx::AiOnnx::OpSet9::Not, args, {}, name)[0];
}

TensorId Builder::logical_or(const std::vector<TensorId> &args,
                             const std::string &name) {
  return impl_->op(Onnx::AiOnnx::OpSet9::Or, args, {}, name)[0];
}

TensorId Builder::pow(const std::vector<TensorId> &args,
                      const std::string &name) {
  return impl_->op(Onnx::AiOnnx::OpSet9::Pow, args, {}, name)[0];
}

TensorId Builder::reciprocal(const std::vector<TensorId> &args,
                             const std::string &name) {
  return impl_->op(Onnx::AiOnnx::OpSet9::Reciprocal, args, {}, name)[0];
}

TensorId Builder::relu(const std::vector<TensorId> &args,
                       const std::string &name) {
  return impl_->op(Onnx::AiOnnx::OpSet9::Relu, args, {}, name)[0];
}

TensorId Builder::scatter(const std::vector<TensorId> args,
                          int64_t axis,
                          const std::string &name) {
  return impl_->op(
      Onnx::AiOnnx::OpSet9::Scatter, args, {{"axis", axis}}, name)[0];
}

TensorId Builder::sigmoid(const std::vector<TensorId> &args,
                          const std::string &name) {
  return impl_->op(Onnx::AiOnnx::OpSet9::Sigmoid, args, {}, name)[0];
}

TensorId Builder::sin(const std::vector<TensorId> &args,
                      const std::string &name) {
  return impl_->op(Onnx::AiOnnx::OpSet9::Sin, args, {}, name)[0];
}

TensorId Builder::sinh(const std::vector<TensorId> &args,
                       const std::string &name) {
  return impl_->op(Onnx::AiOnnx::OpSet9::Sinh, args, {}, name)[0];
}

TensorId Builder::softsign(const std::vector<TensorId> &args,
                           const std::string &name) {
  return impl_->op(Onnx::AiOnnx::OpSet9::Softsign, args, {}, name)[0];
}

TensorId Builder::sqrt(const std::vector<TensorId> &args,
                       const std::string &name) {
  return impl_->op(Onnx::AiOnnx::OpSet9::Sqrt, args, {}, name)[0];
}

TensorId Builder::squeeze(const std::vector<TensorId> &args,
                          const std::vector<int64_t> axes,
                          const std::string &name) {
  return impl_->op(
      Onnx::AiOnnx::OpSet9::Squeeze, args, {{"axes", axes}}, name)[0];
}

TensorId Builder::unsqueeze(const std::vector<TensorId> &args,
                            const std::vector<int64_t> axes,
                            const std::string &name) {
  return impl_->op(
      Onnx::AiOnnx::OpSet9::Unsqueeze, args, {{"axes", axes}}, name)[0];
}

TensorId Builder::sub(const std::vector<TensorId> &args,
                      const std::string &name) {
  return impl_->op(Onnx::AiOnnx::OpSet9::Sub, args, {}, name)[0];
}

TensorId Builder::sum(const std::vector<TensorId> &args,
                      const std::string &name) {
  return impl_->op(Onnx::AiOnnx::OpSet9::Sum, args, {}, name)[0];
}

TensorId Builder::tan(const std::vector<TensorId> &args,
                      const std::string &name) {
  return impl_->op(Onnx::AiOnnx::OpSet9::Tan, args, {}, name)[0];
}

TensorId Builder::tanh(const std::vector<TensorId> &args,
                       const std::string &name) {
  return impl_->op(Onnx::AiOnnx::OpSet9::Tanh, args, {}, name)[0];
}

TensorId Builder::logical_xor(const std::vector<TensorId> &args,
                              const std::string &name) {
  return impl_->op(Onnx::AiOnnx::OpSet9::Xor, args, {}, name)[0];
}

void Builder::verifyWindowParameters(TensorId input,
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

TensorId Builder::convolution(const std::vector<TensorId> &args,
                              const std::vector<int64_t> strides,
                              const std::vector<int64_t> padding,
                              const std::vector<int64_t> dilation,
                              int64_t groups,
                              bool cacheOperation,
                              const std::string &name) {
  return impl_->op(
      Onnx::AiOnnx::OpSet9::Conv,
      args,
      {{"dilations", dilation},
       {"group", groups},
       {"pads", padding},
       {"strides", strides},
       {"__cache_operation", cacheOperation}},
      name,
      [this](std::vector<TensorId> inputs,
             std::map<std::string, boost::any> attributes) {
        this->verifyWindowParameters(
            inputs[0],
            boost::any_cast<const std::vector<int64_t> &>(
                attributes["strides"]),
            boost::any_cast<const std::vector<int64_t> &>(attributes["pads"]),
            boost::any_cast<const std::vector<int64_t> &>(
                attributes["dilations"]));
      })[0];
}

TensorId Builder::averagepool(const std::vector<TensorId> &args,
                              const std::vector<int64_t> kernel_shape,
                              const std::vector<int64_t> strides,
                              const std::vector<int64_t> padding,
                              const std::string &name) {

  return impl_->op(
      Onnx::AiOnnx::OpSet9::AveragePool,
      args,
      {{"count_include_pad", 0},
       {"kernel_shape", kernel_shape},
       {"pads", padding},
       {"strides", strides}},
      name,
      [this](std::vector<TensorId> inputs,
             std::map<std::string, boost::any> attributes) {
        this->verifyWindowParameters(
            inputs[0],
            boost::any_cast<const std::vector<int64_t> &>(
                attributes["strides"]),
            boost::any_cast<const std::vector<int64_t> &>(attributes["pads"]));
      })[0];
}

TensorId Builder::maxpool(const std::vector<TensorId> &args,
                          const std::vector<int64_t> kernel_shape,
                          const std::vector<int64_t> strides,
                          const std::vector<int64_t> padding,
                          const std::string &name) {
  return impl_->op(
      Onnx::AiOnnx::OpSet9::MaxPool,
      args,
      {{"storage_order", 0},
       {"kernel_shape", kernel_shape},
       {"pads", padding},
       {"strides", strides}},
      name,
      [this](std::vector<TensorId> inputs,
             std::map<std::string, boost::any> attributes) {
        this->verifyWindowParameters(
            inputs[0],
            boost::any_cast<const std::vector<int64_t> &>(
                attributes["strides"]),
            boost::any_cast<const std::vector<int64_t> &>(attributes["pads"]));
      })[0];
}

std::tuple<TensorId, TensorId, TensorId>
Builder::lstm(const std::vector<TensorId> &args, const std::string &name) {

  std::vector<TensorId> outputs =
      impl_->op(Onnx::AiOnnx::OpSet9::LSTM, args, {}, name);
  return {outputs[0], outputs[1], outputs[2]};
}

TensorId Builder::gemm(const std::vector<TensorId> &args,
                       float alpha,
                       float beta,
                       int64_t transA,
                       int64_t transB,
                       const std::string &name) {
  return impl_->op(Onnx::AiOnnx::OpSet9::Gemm,
                   args,
                   {{"alpha", alpha},
                    {"beta", beta},
                    {"transA", transA},
                    {"transB", transB}},
                   name)[0];
}

TensorId Builder::pad(const std::vector<TensorId> &args,
                      std::string mode,
                      const std::vector<int64_t> pads,
                      float value,
                      const std::string &name) {

  return impl_->op(
      Onnx::AiOnnx::OpSet9::Pad,
      args,
      {{"mode", mode}, {"pads", pads}, {"value", value}},
      name,
      [this, pads](std::vector<TensorId> inputs,
                   std::map<std::string, boost::any>) {
        auto rank = getTensorShape(inputs[0]).size();
        if (pads.size() != rank * 2) {
          throw error(
              "Padding vector (length {}) doesn't contain 2 entries per input "
              "dimension {}",
              pads.size(),
              rank);
        }
      })[0];
}

TensorId Builder::matmul(const std::vector<TensorId> &args,
                         const std::string &name) {
  return impl_->op(Onnx::AiOnnx::OpSet9::MatMul, args, {}, name)[0];
}

TensorId Builder::shape(const std::vector<TensorId> &args,
                        const std::string &name) {
  return impl_->op(Onnx::AiOnnx::OpSet9::Shape, args, {}, name)[0];
}

TensorId Builder::slice(const std::vector<TensorId> &args,
                        const std::vector<int64_t> &axes,
                        const std::vector<int64_t> &starts,
                        const std::vector<int64_t> &ends,
                        const std::string &name) {

  return impl_->op(Onnx::AiOnnx::OpSet9::Slice,
                   args,
                   {{"axes", axes}, {"starts", starts}, {"ends", ends}},
                   name)[0];
}

TensorId Builder::softmax(const std::vector<TensorId> &args,
                          const std::string &name) {
  return impl_->op(Onnx::AiOnnx::OpSet9::Softmax, args, {{"axes", 1}}, name)[0];
}

TensorId Builder::subsample(const std::vector<TensorId> &args,
                            const std::vector<int64_t> &strides,
                            const std::string &name) {

  for (int i = 0; i < strides.size(); ++i) {
    if (strides[i] == 0)
      throw error("Strides invalid. 0 stride at index {}", i);
  }

  return impl_->op(Onnx::AiGraphcore::OpSet1::Subsample,
                   args,
                   {{"strides", strides}},
                   name)[0];
}

Builder::BatchNormalizationTrainingOutputs
Builder::batchnormalizationTraining(const TensorId x,
                                    const TensorId scale,
                                    const TensorId b,
                                    const TensorId mean,
                                    const TensorId var,
                                    const float epsilon,
                                    const float momentum,
                                    const int spatial,
                                    const std::string &name) {

  Builder::BatchNormalizationTrainingOutputs bnOuputs;

  std::vector<TensorId> outputs = impl_->op(
      Onnx::AiOnnx::OpSet9::BatchNormalization,
      {x, scale, b, mean, var},
      {{"epsilon", epsilon}, {"momentum", momentum}, {"spatial", spatial}},
      name);

  bnOuputs.y         = outputs[0];
  bnOuputs.mean      = outputs[1];
  bnOuputs.var       = outputs[2];
  bnOuputs.savedMean = outputs[3];
  bnOuputs.savedVar  = outputs[4];

  return bnOuputs;
}

TensorId Builder::batchnormalizationTesting(const TensorId x,
                                            const TensorId scale,
                                            const TensorId b,
                                            const TensorId mean,
                                            const TensorId var,
                                            const float epsilon,
                                            const float momentum,
                                            const int spatial,
                                            const std::string &name) {

  return impl_->op(
      Onnx::AiOnnx::OpSet9::BatchNormalization,
      {x, scale, b, mean, var},
      1,
      {{"epsilon", epsilon}, {"momentum", momentum}, {"spatial", spatial}},
      name)[0];
}

TensorId Builder::transpose(const std::vector<TensorId> &args,
                            const std::vector<int64_t> &perm,
                            const std::string &name) {
  return impl_->op(
      Onnx::AiOnnx::OpSet9::Transpose, args, {{"perm", perm}}, name)[0];
}

TensorId Builder::reshape(const std::vector<TensorId> &args,
                          const std::string &name) {
  return impl_->op(Onnx::AiOnnx::OpSet9::Reshape, args, {}, name)[0];
}

TensorId Builder::reshape_const(const std::vector<TensorId> &args,
                                const std::vector<int64_t> &shape,
                                const std::string &name) {
  return impl_->reshape_const(args, shape, name);
}

std::vector<TensorId> Builder::customOp(
    const OperatorIdentifier &opid,
    const std::vector<boost::any> &inputs,
    const unsigned numOutputs,
    const std::vector<std::pair<std::string, boost::any>> &attributes,
    const std::string &name) {
  return impl_->customOp(opid, inputs, numOutputs, attributes, name);
}

void Builder::addNodeAttribute(const std::string &attributeName,
                               const int64_t &attributeValue,
                               const std::set<TensorId> &nodeOutputNames) {
  impl_->addNodeAttribute(attributeName, attributeValue, nodeOutputNames);
}

void Builder::addNodeAttribute(const std::string &attributeName,
                               const std::vector<int64_t> &attributeValue,
                               const std::set<TensorId> &nodeOutputNames) {
  impl_->addNodeAttribute(attributeName, attributeValue, nodeOutputNames);
}

void Builder::addNodeAttribute(const std::string &attributeName,
                               const float &attributeValue,
                               const std::set<TensorId> &nodeOutputNames) {
  impl_->addNodeAttribute(attributeName, attributeValue, nodeOutputNames);
}

void Builder::addNodeAttribute(const std::string &attributeName,
                               const std::vector<float> &attributeValue,
                               const std::set<TensorId> &nodeOutputNames) {
  impl_->addNodeAttribute(attributeName, attributeValue, nodeOutputNames);
}

void Builder::addNodeAttribute(const std::string &attributeName,
                               const std::string &attributeValue,
                               const std::set<TensorId> &nodeOutputNames) {
  impl_->addNodeAttribute(attributeName, attributeValue, nodeOutputNames);
}

void Builder::addNodeAttribute(const std::string &attributeName,
                               const char *attributeValue,
                               const std::set<TensorId> &nodeOutputNames) {
  impl_->addNodeAttribute(attributeName, attributeValue, nodeOutputNames);
}

void Builder::addNodeAttribute(const std::string &attributeName,
                               const std::vector<std::string> &attributeValue,
                               const std::set<TensorId> &nodeOutputNames) {
  impl_->addNodeAttribute(attributeName, attributeValue, nodeOutputNames);
}

void Builder::addNodeAttribute(const std::string &attributeName,
                               const bool attributeValue,
                               const std::set<TensorId> &nodeOutputNames) {
  impl_->addNodeAttribute(attributeName, attributeValue, nodeOutputNames);
}

void Builder::addNodeAttribute(const std::string &attributeName,
                               const ConstVoidData &attributeValue,
                               const std::set<TensorId> &nodeOutputNames) {
  impl_->addNodeAttribute(attributeName, attributeValue, nodeOutputNames);
}

bool Builder::nodeHasAttribute(const std::string &attributeName,
                               const std::set<TensorId> &nodeOutputNames) {
  return impl_->nodeHasAttribute(attributeName, nodeOutputNames);
}

int64_t
Builder::getInt64NodeAttribute(const std::string &attributeName,
                               const std::set<TensorId> &nodeOutputNames) {
  return impl_->getInt64NodeAttribute(attributeName, nodeOutputNames);
}

std::vector<int64_t> Builder::getInt64VectorNodeAttribute(
    const std::string &attributeName,
    const std::set<TensorId> &nodeOutputNames) {
  return impl_->getInt64VectorNodeAttribute(attributeName, nodeOutputNames);
}

float Builder::getFloatNodeAttribute(
    const std::string &attributeName,
    const std::set<TensorId> &nodeOutputNames) {
  return impl_->getFloatNodeAttribute(attributeName, nodeOutputNames);
}

std::vector<float> Builder::getFloatVectorNodeAttribute(
    const std::string &attributeName,
    const std::set<TensorId> &nodeOutputNames) {
  return impl_->getFloatVectorNodeAttribute(attributeName, nodeOutputNames);
}

std::string
Builder::getStringNodeAttribute(const std::string &attributeName,
                                const std::set<TensorId> &nodeOutputNames) {
  return impl_->getStringNodeAttribute(attributeName, nodeOutputNames);
}

std::vector<std::string> Builder::getStringVectorNodeAttribute(
    const std::string &attributeName,
    const std::set<TensorId> &nodeOutputNames) {
  return impl_->getStringVectorNodeAttribute(attributeName, nodeOutputNames);
}

bool Builder::getBoolNodeAttribute(const std::string &attributeName,
                                   const std::set<TensorId> &nodeOutputNames) {
  return impl_->getBoolNodeAttribute(attributeName, nodeOutputNames);
}

void Builder::removeNodeAttribute(const std::string &attributeName,
                                  const std::set<TensorId> &nodeOutputNames) {
  impl_->removeNodeAttribute(attributeName, nodeOutputNames);
}

std::vector<std::string>
Builder::getAllNodeAttributeNames(const std::set<TensorId> &nodeOutputNames) {
  return impl_->getAllNodeAttributeNames(nodeOutputNames);
}

void Builder::loadModelProto(const std::string &modelProtoOrFilename) {
  impl_->loadModelProto(modelProtoOrFilename);
}

std::string Builder::getModelProto() const { return impl_->getModelProto(); }

std::vector<TensorId> Builder::getInputTensorIds() const {
  return impl_->getInputTensorIds();
}

std::vector<TensorId> Builder::getOutputTensorIds() const {
  return impl_->getOutputTensorIds();
}

std::vector<TensorId> Builder::getValueTensorIds() const {
  return impl_->getValueTensorIds();
}

std::vector<int64_t> Builder::getTensorShape(const TensorId id) {
  return impl_->getTensorShape(id);
}

void Builder::convertInitializersToConstants(const std::vector<TensorId> &ids) {
  impl_->convertInitializersToConstants(ids);
}

void Builder::convertAllFixedPointInitializersToConstants() {
  impl_->convertAllFixedPointInitializersToConstants();
}

void Builder::setAttribute(const std::string &attribute, boost::any value) {
  impl_->setAttribute(attribute, value);
}

void Builder::clearAttribute(const std::string &attribute) {
  impl_->clearAttribute(attribute);
}

} // namespace poponnx
