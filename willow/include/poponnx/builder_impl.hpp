#ifndef GUARD_BUILDER_IMPL_H
#define GUARD_BUILDER_IMPL_H

#include <poponnx/builder.hpp>
#include <poponnx/names.hpp>

#include <map>
#include <string>

namespace willow {

/**
 * An implementation of a Builder
 */
class BuilderImpl {
public:
  BuilderImpl();

  void configure();

  TensorId addInputTensor(const TensorInfo &tensorInfo);
  TensorId addInitializedInputTensor(const ConstVoidData &initData);

  void addOutputTensor(const TensorId &arg0);

  // Operations requiring only tensor inputs
  TensorId abs(const std::vector<TensorId> &args);
  TensorId acos(const std::vector<TensorId> &args);
  TensorId acosh(const std::vector<TensorId> &args);
  TensorId add(const std::vector<TensorId> &args);
  TensorId logical_and(const std::vector<TensorId> &args);
  TensorId asin(const std::vector<TensorId> &args);
  TensorId asinh(const std::vector<TensorId> &args);
  TensorId atan(const std::vector<TensorId> &args);
  TensorId atanh(const std::vector<TensorId> &args);
  TensorId cast(const std::vector<TensorId> &args);
  TensorId ceil(const std::vector<TensorId> &args);
  TensorId cos(const std::vector<TensorId> &args);
  TensorId cosh(const std::vector<TensorId> &args);
  TensorId div(const std::vector<TensorId> &args);
  TensorId elu(const std::vector<TensorId> &args);
  TensorId equal(const std::vector<TensorId> &args);
  TensorId exp(const std::vector<TensorId> &args);
  TensorId floor(const std::vector<TensorId> &args);
  TensorId greater(const std::vector<TensorId> &args);
  TensorId identity(const std::vector<TensorId> &args);
  TensorId less(const std::vector<TensorId> &args);
  TensorId log(const std::vector<TensorId> &args);
  TensorId max(const std::vector<TensorId> &args);
  TensorId mean(const std::vector<TensorId> &args);
  TensorId min(const std::vector<TensorId> &args);
  TensorId mul(const std::vector<TensorId> &args);
  TensorId neg(const std::vector<TensorId> &args);
  TensorId logical_not(const std::vector<TensorId> &args);
  TensorId logical_or(const std::vector<TensorId> &args);
  TensorId pow(const std::vector<TensorId> &args);
  TensorId reciprocal(const std::vector<TensorId> &args);
  TensorId relu(const std::vector<TensorId> &args);
  TensorId sigmoid(const std::vector<TensorId> &args);
  TensorId sin(const std::vector<TensorId> &args);
  TensorId sinh(const std::vector<TensorId> &args);
  TensorId softsign(const std::vector<TensorId> &args);
  TensorId sqrt(const std::vector<TensorId> &args);
  TensorId sub(const std::vector<TensorId> &args);
  TensorId sum(const std::vector<TensorId> &args);
  TensorId tan(const std::vector<TensorId> &args);
  TensorId tanh(const std::vector<TensorId> &args);
  TensorId logical_xor(const std::vector<TensorId> &args);

  TensorId convolution(const std::vector<TensorId> &args,
                       const std::vector<int64_t> strides,
                       const std::vector<int64_t> padding,
                       const std::vector<int64_t> dilation,
                       int64_t groups      = 1,
                       bool cacheOperation = true);

  TensorId averagepool(const std::vector<TensorId> &args,
                       const std::vector<int64_t> kernel_shape,
                       const std::vector<int64_t> strides,
                       const std::vector<int64_t> padding);

  TensorId maxpool(const std::vector<TensorId> &args,
                   const std::vector<int64_t> kernel_shape,
                   const std::vector<int64_t> strides,
                   const std::vector<int64_t> padding);

  TensorId gemm(const std::vector<TensorId> &args,
                float alpha,
                float beta,
                int64_t transA,
                int64_t transB);

  TensorId matmul(const std::vector<TensorId> &args);

  void addNodeAttribute(const std::string &attributeName,
                        const int64_t &attributeValue,
                        const std::set<TensorId> &nodeOutputNames);

  void addNodeAttribute(const std::string &attributeName,
                        const std::vector<int64_t> &attributeValue,
                        const std::set<TensorId> &nodeOutputNames);

  void addNodeAttribute(const std::string &attributeName,
                        const float &attributeValue,
                        const std::set<TensorId> &nodeOutputNames);

  void addNodeAttribute(const std::string &attributeName,
                        const std::vector<float> &attributeValue,
                        const std::set<TensorId> &nodeOutputNames);

  void addNodeAttribute(const std::string &attributeName,
                        const std::string &attributeValue,
                        const std::set<TensorId> &nodeOutputNames);

  void addNodeAttribute(const std::string &attributeName,
                        const std::vector<std::string> &attributeValue,
                        const std::set<TensorId> &nodeOutputNames);

  bool nodeHasAttribute(const std::string &attributeName,
                        const std::set<TensorId> &nodeOutputNames);

  int64_t getInt64NodeAttribute(const std::string &attributeName,
                                const std::set<TensorId> &nodeOutputNames);

  std::vector<int64_t>
  getInt64VectorNodeAttribute(const std::string &attributeName,
                              const std::set<TensorId> &nodeOutputNames);

  float getFloatNodeAttribute(const std::string &attributeName,
                              const std::set<TensorId> &nodeOutputNames);

  std::vector<float>
  getFloatVectorNodeAttribute(const std::string &attributeName,
                              const std::set<TensorId> &nodeOutputNames);

  std::string getStringNodeAttribute(const std::string &attributeName,
                                     const std::set<TensorId> &nodeOutputNames);

  std::vector<std::string>
  getStringVectorNodeAttribute(const std::string &attributeName,
                               const std::set<TensorId> &nodeOutputNames);

  void removeNodeAttribute(const std::string &attributeName,
                           const std::set<TensorId> &nodeOutputNames);

  std::vector<std::string>
  getAllNodeAttributeNames(const std::set<TensorId> &nodeOutputNames);

  void loadModelProto(const std::string &modelProtoOrFilename);

  const std::map<std::string, TensorId> getTensorTranslation() const;

  std::string getModelProto() const;

  std::vector<TensorId> getInputTensorIds() const;

  std::vector<TensorId> getOutputTensorIds() const;

  std::vector<int64_t> getTensorShape(const TensorId id);

private:
  TensorId add_simple_op(const std::vector<TensorId> &args,
                         const char *name,
                         int arg_count);

  TensorId add_variadic_op(const std::vector<TensorId> &args, const char *name);

  TensorId getNextId();

  void uniquifyNames(onnx::GraphProto &graph);

  bool isInputTensor(TensorId id) const;

  bool isOutputTensor(TensorId id) const;

  std::string getStrFromTensorIdVec(std::vector<TensorId> v) const;

  int getInputTensorIndex(TensorId id) const;

  int getOutputTensorIndex(TensorId id) const;

  const onnx::ValueInfoProto &getValueInfoProto(TensorId id) const;

  bool
  findNodeProtoByOutputNamesImpl(onnx::NodeProto *&out,
                                 const std::set<TensorId> &nodeOutputNames);

  onnx::NodeProto &
  findNodeProtoByOutputNames(const std::set<TensorId> &nodeOutputNames);

  bool nodeHasAttributeImpl(onnx::AttributeProto *&out,
                            onnx::NodeProto &node,
                            const std::string &attributeName);

  onnx::AttributeProto &
  addNewAttributeToNode(const std::string &attributeName,
                        const std::set<TensorId> &nodeOutputNames);

  onnx::AttributeProto &
  getNodeAttribute(const std::string &attributeName,
                   const std::set<TensorId> &nodeOutputNames);

  uint64_t next_id_ = 0;

  onnx::ModelProto model_;

  std::map<std::string, TensorId> tensorTranslation_;
};

} // namespace willow
#endif // GUARD_BUILDER_IMPL_H
