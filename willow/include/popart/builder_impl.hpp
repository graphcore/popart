#ifndef GUARD_BUILDER_IMPL_HPP
#define GUARD_BUILDER_IMPL_HPP

#include <map>
#include <string>
#include <popart/builder.hpp>
#include <popart/names.hpp>
#include <popart/opidentifier.hpp>

#include <boost/any.hpp>

// The BuilderImpl class has an onnx::ModelProto, so we cannot
// use the forward declarations in names.hpp at this point
#include <onnx/onnx_pb.h>

namespace onnx {
class NodeProto;
}

namespace popart {

/**
 * An implementation of a Builder
 */
class BuilderImpl {
public:
  BuilderImpl() = default;

  void configure();

  TensorId addInputTensor(const TensorInfo &tensorInfo,
                          const std::string &debugPrefix = "");

  void addInputTensorFromHigherScope(const TensorId &tensorId);

  TensorId addInitializedInputTensor(const ConstVoidData &initData,
                                     const std::string &debugPrefix = "");

  void addOutputTensor(const TensorId &arg0);

  /**
   * Add an op to the model
   *
   *
   * \param opid The operator identifier
   * \param opsetVersion The opset for the domain of the op
   * \param inputs The input tensor ids
   * \param outputs The output tensor ids
   * \param opAttributes The attributes of the op
   * \param name Debug name
   * \param validateInput Callback function to validate the inputs & attributes
   */

  void op(const OperatorIdentifier &opid,
          int opsetVersion,
          const std::vector<TensorId> &inputs,
          const std::vector<TensorId> &outputs,
          const std::map<std::string, boost::any> &opAttributes,
          const std::string &name,
          std::function<void(std::vector<TensorId>,
                             std::map<std::string, boost::any>)> validateInput =
              nullptr);

  /**
   * Add an op to the model
   *
   *
   * \param opid The operator identifier
   * \param opsetVersion The opset for the domain of the op
   * \param inputs The input tensor ids
   * \param opAttributes The attributes of the op
   * \param name Debug name
   * \param validateInput Callback function to validate the inputs & attributes
   * \return A list of output tensor ids. Size is given by numberOfOutputs
   */

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

  /**
   * Add an op to the model
   *
   *
   * \param opid The operator identifier
   * \param opsetVersion The opset for the domain of the op
   * \param inputs The input tensor ids
   * \param numOutputs The number if output tensors
   * \param opAttributes The attributes of the op
   * \param name Debug name
   * \param validateInput Callback function to validate the inputs & attributes
   * \return A list of output tensor ids. Size is given by numberOfOutputs
   */

  std::vector<TensorId>
  op(const OperatorIdentifier &opid,
     int opsetVersion,
     const std::vector<TensorId> &inputs,
     unsigned int numOutputs,
     const std::map<std::string, boost::any> &opAttributes,
     const std::string &name,
     std::function<void(std::vector<TensorId>,
                        std::map<std::string, boost::any>)> validateInput =
         nullptr) {

    std::vector<TensorId> outputs(numOutputs);

    // Generate the output names
    for (int i = 0; i < numOutputs; ++i) {
      outputs[i] = getNextId(opid.type, i);
    }

    op(opid, opsetVersion, inputs, outputs, opAttributes, name, validateInput);

    return outputs;
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

  std::vector<TensorId> getInputTensorIds() const;

  std::vector<TensorId> getOutputTensorIds() const;

  std::vector<TensorId> getValueTensorIds() const;

  std::vector<int64_t> getTensorShape(const TensorId id);

  void setAttribute(const std::string &attribute, boost::any value);
  void clearAttribute(const std::string &attribute);

  void pushNameScope(const std::string &name);
  void popNameScope();
  std::string getNameScope(const std::string &name = "") const;

  const BuilderImpl *getParent() const;
  bool hasParent() const { return nullptr != parent; }
  std::vector<const BuilderImpl *> getChildren() const;

  onnx::NodeProto &
  findNodeProtoByOutputNames(const std::set<TensorId> &nodeOutputNames);

private:
  void finalizeOp(onnx::NodeProto *node, const std::string &name);

  void addOpsetRequirement(const std::string &domain, int version);

  TensorId getNextId(const std::string &name, int n = -1);

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

  std::vector<std::string> name_scope_stack_;

  // Local set of TensorIds created in the
  // scope of this Builder
  std::set<TensorId> tensorIds;

  bool inCurrentScope(const TensorId &) const;

  // in parent's scope, or higher
  bool inHigherScope(const TensorId &) const;

  // in a child's scope, or lower
  bool inLowerScope(const TensorId &) const;

  onnx::ModelProto model_;

  std::map<std::string, boost::any> attributes;

  // Record which opset version we are using for each domain
  std::map<std::string, int64_t> opsetVersions;

  const BuilderImpl *parent{nullptr};
  std::vector<const BuilderImpl *> children;

public:
  void addChild(const BuilderImpl *child) { children.push_back(child); }

  void setParent(const BuilderImpl *parent_) { parent = parent_; }
};

} // namespace popart

#endif // GUARD_BUILDER_IMPL_HPP
