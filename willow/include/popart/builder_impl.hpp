// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_BUILDER_IMPL_HPP
#define GUARD_BUILDER_IMPL_HPP

#include <map>
#include <string>
#include <popart/builder.hpp>
#include <popart/names.hpp>
#include <popart/opidentifier.hpp>

#include <popart/vendored/any.hpp>

// The BuilderImpl class has an ONNX_NAMESPACE::ModelProto, so we cannot
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
                          const popart::DebugContext &debugContext = {});

  TensorId addUntypedInputTensor(const popart::DebugContext &debugContext = {});

  void addInputTensorFromParentGraph(const TensorId &tensorId);

  TensorId
  addInitializedInputTensor(const ConstVoidData &initData,
                            const popart::DebugContext &debugContext = {});

  void addOutputTensor(const TensorId &arg0);

  void setGraphName(const std::string &name);

  /**
   * Add an op to the model.
   *
   *
   * \param opid The operator identifier.
   * \param opsetVersion The opset for the domain of the op.
   * \param inputs The input tensor ids.
   * \param outputs The output tensor ids.
   * \param opAttributes The attributes of the op.
   * \param debugContext Debug context.
   * \param validateInput Callback function to validate the inputs & attributes.
   */

  void op(const OperatorIdentifier &opid,
          int opsetVersion,
          const std::vector<TensorId> &inputs,
          const std::vector<TensorId> &outputs,
          const std::map<std::string, popart::any> &opAttributes,
          const DebugContext &debugContext,
          std::function<void(std::vector<TensorId>,
                             std::map<std::string, popart::any>)>
              validateInput = nullptr);

  /**
   * Add an op to the model.
   *
   *
   * \param opid The operator identifier.
   * \param opsetVersion The opset for the domain of the op.
   * \param inputs The input tensor ids.
   * \param opAttributes The attributes of the op.
   * \param debugContext Debug context.
   * \param validateInput Callback function to validate the inputs & attributes.
   * \return A list of output tensor ids. Size is given by \c numberOfOutputs.
   */

  std::vector<TensorId>
  op(const OperatorIdentifier &opid,
     int opsetVersion,
     const std::vector<TensorId> &inputs,
     const std::map<std::string, popart::any> &opAttributes,
     const DebugContext &debugContext,
     std::function<void(std::vector<TensorId>,
                        std::map<std::string, popart::any>)> validateInput =
         nullptr) {
    return op(opid,
              opsetVersion,
              inputs,
              opid.numOutputs,
              opAttributes,
              debugContext,
              validateInput);
  }

  /**
   * Add an op to the model.
   *
   *
   * \param opid The operator identifier.
   * \param opsetVersion The opset for the domain of the op.
   * \param inputs The input tensor ids.
   * \param numOutputs The number if output tensors.
   * \param opAttributes The attributes of the op.
   * \param name Debug information.
   * \param validateInput Callback function to validate the inputs & attributes.
   * \return A list of output tensor ids. Size is given by \c numberOfOutputs.
   */

  std::vector<TensorId>
  op(const OperatorIdentifier &opid,
     int opsetVersion,
     const std::vector<TensorId> &inputs,
     unsigned int numOutputs,
     const std::map<std::string, popart::any> &opAttributes,
     const DebugContext &debugContext,
     std::function<void(std::vector<TensorId>,
                        std::map<std::string, popart::any>)> validateInput =
         nullptr) {

    std::vector<TensorId> outputs(numOutputs);

    // Generate the output names
    for (int i = 0; i < numOutputs; ++i) {
      outputs[i] = getNextId(opid.type, i);
    }

    op(opid,
       opsetVersion,
       inputs,
       outputs,
       opAttributes,
       debugContext,
       validateInput);

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

  void saveModelProto(const std::string &fn);

  // Note: The onnx external_data_helper.py has the same functionality in its
  // convert_model_to_external_data method. However, this assumes a
  // TensorProto's data is only stored as raw_data, when it is valid to store
  // data in any 'data' field. Here we do not make this assumption.
  void saveInitializersExternally(const std::vector<TensorId> &ids,
                                  const std::string &fn);

  std::string getModelProto() const;

  std::vector<TensorId> getInputTensorIds() const;

  std::vector<TensorId> getOutputTensorIds() const;

  std::vector<TensorId> getValueTensorIds() const;

  std::vector<TensorId> getTrainableTensorIds() const;

  std::set<TensorId> getValidInputTensorIds() const;

  bool isInputTensor(const TensorId &id) const;

  bool isOutputTensor(const TensorId &id) const;

  bool isValueTensor(const TensorId &id) const;

  bool hasTensorShape(const TensorId &id) const;

  std::vector<int64_t> getTensorShape(const TensorId &id);

  std::string getTensorDtypeString(const TensorId &id);

  DataType getTensorDataType(const TensorId &id);

  bool isInitializer(const TensorId &id) const;

  void setAttribute(const std::string &attribute, popart::any value);
  popart::any getAttribute(const std::string &attribute) const;
  bool hasAttribute(const std::string &attribute) const;
  void clearAttribute(const std::string &attribute);
  bool hasAttribute(const std::string &attribute);
  popart::any getAttribute(const std::string &attribute);

  void pushNameScope(const std::string &name);
  void popNameScope();
  std::string getNameScope(const std::string &name = "") const;

  const BuilderImpl *getParent() const;
  bool hasParent() const { return nullptr != parent; }
  std::vector<const BuilderImpl *> getChildren() const;

  ONNX_NAMESPACE::NodeProto &
  findNodeProtoByOutputNames(const std::set<TensorId> &nodeOutputNames);

  static void
  populateTensorProtoFromConstVoidData(const ConstVoidData &initData,
                                       const std::string &id,
                                       ONNX_NAMESPACE::TensorProto *tp);

  std::vector<TensorId>
  checkpointOutput(const std::vector<TensorId> &nodeOutputNames);

private:
  ONNX_NAMESPACE::ValueInfoProto *addGraphInput(const TensorId &id);

  void finalizeOp(ONNX_NAMESPACE::NodeProto *node,
                  const OperatorIdentifier &,
                  const std::string &name);

  void runShapeInference(ONNX_NAMESPACE::NodeProto *node,
                         const OperatorIdentifier &);

  void addOpsetRequirement(const std::string &domain, int version);

  TensorId getNextId(const std::string &name, int n = -1);
  TensorId getNextInputId(const std::string &debugPrefix);

  std::string getStrFromTensorIdVec(std::vector<TensorId> v) const;

  int getInputTensorIndex(TensorId id) const;

  int getOutputTensorIndex(TensorId id) const;

  int getValueTensorIndex(TensorId id) const;

  const ONNX_NAMESPACE::ValueInfoProto &getValueInfoProto(TensorId id) const;

  bool
  findNodeProtoByOutputNamesImpl(ONNX_NAMESPACE::NodeProto *&out,
                                 const std::set<TensorId> &nodeOutputNames);

  bool nodeHasAttributeImpl(ONNX_NAMESPACE::AttributeProto *&out,
                            ONNX_NAMESPACE::NodeProto &node,
                            const std::string &attributeName);

  ONNX_NAMESPACE::AttributeProto &
  addNewAttributeToNode(const std::string &attributeName,
                        const std::set<TensorId> &nodeOutputNames);

  ONNX_NAMESPACE::AttributeProto &
  addNewAttributeToNode(const std::string &attributeName,
                        ONNX_NAMESPACE::NodeProto &node);

  ONNX_NAMESPACE::AttributeProto &
  getNodeAttribute(const std::string &attributeName,
                   const std::set<TensorId> &nodeOutputNames);

  void addNodeAttribute(const std::string &attributeName,
                        const popart::any &attributeValue,
                        ONNX_NAMESPACE::NodeProto &node);

  std::vector<std::string> name_scope_stack_;

  // Local set of TensorIds created in the
  // scope of this Builder
  std::set<TensorId> tensorIds;

  bool inCurrentScope(const TensorId &) const;

  // in parent's scope, or higher
  bool inHigherScope(const TensorId &) const;

  // in a child's scope, or lower
  bool inLowerScope(const TensorId &) const;

  ONNX_NAMESPACE::ModelProto model_;

  std::map<std::string, popart::any> attributes;

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
