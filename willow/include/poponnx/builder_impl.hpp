#ifndef GUARD_BUILDER_IMPL_H
#define GUARD_BUILDER_IMPL_H

#include <map>
#include <string>
#include <poponnx/builder.hpp>
#include <poponnx/names.hpp>
#include <poponnx/opidentifier.hpp>

#include <boost/any.hpp>

// The BuilderImpl class has an onnx::ModelProto, so we cannot
// use the forward declarations in names.hpp at this point
#include <onnx/onnx_pb.h>

namespace onnx {
class NodeProto;
}

namespace poponnx {

/**
 * An implementation of a Builder
 */
class BuilderImpl {
public:
  BuilderImpl();

  enum class ExecutionMode { INFERENCE = 0, TRAINING };

  static std::vector<std::string>
  listConstExprNodesModel(const onnx::ModelProto &model, ExecutionMode mode);

  static std::vector<std::string>
  listNonConstExprNodesModel(const onnx::ModelProto &model, ExecutionMode mode);

  void configure();

  TensorId addInputTensor(const TensorInfo &tensorInfo);
  TensorId addInitializedInputTensor(const ConstVoidData &initData);

  void addOutputTensor(const TensorId &arg0);

  /**
   * Add an op to the model
   *
   *
   * \param opid The operator identifier
   * \param opsetVersion The opset for the domain of the op
   * \param inputs The input tensor ids
   * \param numberOfOutputs The number if output tensors
   * \param opAttributes The attributes of the op
   * \param name Debug name
   * \param validateInput Callback function to validate the inputs & attributes
   * \return A list of output tensor ids. Size is given by numberOfOutputs
   */

  std::vector<TensorId>
  op(const OperatorIdentifier &opid,
     int opsetVersion,
     const std::vector<TensorId> &inputs,
     const unsigned numberOfOutputs,
     const std::map<std::string, boost::any> &opAttributes,
     const std::string &name,
     std::function<void(std::vector<TensorId>,
                        std::map<std::string, boost::any>)> validateInput =
         nullptr);

  std::vector<TensorId>
  op(const OperatorIdentifier &opid,
     int opsetVersion,
     const std::vector<TensorId> &inputs,
     const std::map<std::string, boost::any> &opAttributes,
     const std::string &name,
     std::function<void(std::vector<TensorId>,
                        std::map<std::string, boost::any>)> validateInput =
         nullptr) {
    return op(opid,
              opsetVersion,
              inputs,
              opid.numOutputs,
              opAttributes,
              name,
              validateInput);
  }

  // The following do seem to be ripe for a template

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
                        const char *attributeValue,
                        const std::set<TensorId> &nodeOutputNames);

  void addNodeAttribute(const std::string &attributeName,
                        const std::vector<std::string> &attributeValue,
                        const std::set<TensorId> &nodeOutputNames);

  void addNodeAttribute(const std::string &attributeName,
                        const bool attributeValue,
                        const std::set<TensorId> &nodeOutputNames);

  void addNodeAttribute(const std::string &attributeName,
                        const int &attributeValue,
                        const std::set<TensorId> &nodeOutputNames);

  void addNodeAttribute(const std::string &attributeName,
                        const ConstVoidData &attributeValue,
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

  bool getBoolNodeAttribute(const std::string &attributeName,
                            const std::set<TensorId> &nodeOutputNames);

  void removeNodeAttribute(const std::string &attributeName,
                           const std::set<TensorId> &nodeOutputNames);

  std::vector<std::string>
  getAllNodeAttributeNames(const std::set<TensorId> &nodeOutputNames);

  void loadModelProto(const std::string &modelProtoOrFilename);

  std::string getModelProto() const;

  std::vector<std::string> listConstExprNodes(ExecutionMode mode) const;

  std::vector<std::string> listNonConstExprNodes(ExecutionMode mode) const;

  std::vector<TensorId> getInputTensorIds() const;

  std::vector<TensorId> getOutputTensorIds() const;

  std::vector<TensorId> getValueTensorIds() const;

  std::vector<int64_t> getTensorShape(const TensorId id);

  void setAttribute(const std::string &attribute, boost::any value);
  void clearAttribute(const std::string &attribute);

private:
  void finalizeOp(onnx::NodeProto *node, const std::string &name);

  void addOpsetRequirement(const std::string &domain, int version);

  TensorId getNextId();

  bool isInputTensor(TensorId id) const;

  bool isOutputTensor(TensorId id) const;

  bool isValueTensor(TensorId id) const;

  std::string getStrFromTensorIdVec(std::vector<TensorId> v) const;

  int getInputTensorIndex(TensorId id) const;

  int getOutputTensorIndex(TensorId id) const;

  int getValueTensorIndex(TensorId id) const;

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

  onnx::AttributeProto &addNewAttributeToNode(const std::string &attributeName,
                                              onnx::NodeProto &node);

  onnx::AttributeProto &
  getNodeAttribute(const std::string &attributeName,
                   const std::set<TensorId> &nodeOutputNames);

  void addNodeAttribute(const std::string &attributeName,
                        const boost::any &attributeValue,
                        onnx::NodeProto &node);

  uint64_t next_id_ = 0;

  onnx::ModelProto model_;

  std::map<std::string, boost::any> attributes;

  // Record which opset version we are using for each domain
  std::map<std::string, int64_t> opsetVersions;
};

} // namespace poponnx
#endif // GUARD_BUILDER_IMPL_H
