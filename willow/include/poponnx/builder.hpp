#ifndef GUARD_BUILDER_H
#define GUARD_BUILDER_H

#include <memory>
#include <set>
#include <string>
#include <vector>

#include <poponnx/names.hpp>
#include <poponnx/opidentifier.hpp>
#include <poponnx/tensorinfo.hpp>

#include <boost/any.hpp>
#include <boost/optional.hpp>

namespace poponnx {

class BuilderImpl;
class Builder;
class TensorInfo;
enum class DataType;

class DomainOpSet {

protected:
  std::unique_ptr<BuilderImpl> &impl;

  virtual int getOpsetVersion() const = 0;

public:
  DomainOpSet(std::unique_ptr<BuilderImpl> &impl_) : impl(impl_) {}
  DomainOpSet(const DomainOpSet &other) = default;
  virtual ~DomainOpSet()                = default;
};

// Include the generated builder.h code
#include "builder.h.gen"

class AiOnnxMlOpset1 : public DomainOpSet {

protected:
  using DomainOpSet::impl;

  int getOpsetVersion() const override { return 1; }

public:
  AiOnnxMlOpset1(std::unique_ptr<BuilderImpl> &impl_) : DomainOpSet(impl_) {}
};

class AiGraphcoreOpset1 : public DomainOpSet {

protected:
  using DomainOpSet::impl;

  int getOpsetVersion() const override { return 1; }

public:
  AiGraphcoreOpset1(std::unique_ptr<BuilderImpl> &impl_) : DomainOpSet(impl_) {}

  /**
   * Add a subsample operation to the model
   *
   * This is a poplar extention
   *
   * If multiple tensors are provided that strides will applied to them all
   *
   * \param args Tensor T
   * \param strides The strides
   * \param name Optional identifer for operation
   * \return The name of the result tensor
   */
  TensorId subsample(const std::vector<TensorId> &args,
                     const std::vector<int64_t> &strides,
                     const std::string &name = {});
};

/**
 * An interface for a Builder, used for creating ONNX graphs.
 */
class Builder {
  Builder();

public:
  /**
   * Create a builder for an ONNX model.
   */
  static std::unique_ptr<Builder> create();

  /**
   * Create a builder which loads a serialized ONNX ModelProto into the builder
   * and validates it.
   *
   * \param modelProtoOrFilename Either an ONNX model protobuf, or the name of a
   *                             file containing an ONNX model protobuf.
   */
  static std::unique_ptr<Builder>
  createFromOnnxModel(const std::string &modelProtoOrFilename);

  enum class ExecutionMode { INFERENCE = 0, TRAINING };

  /**
   * Retrieve the set of constexpr nodes
   *
   * \param model The model to get the nodes from
   * \param mode The intended execution mode
   *
   * \return A vector of node names
   */
  static std::vector<std::string>
  listConstExprNodesModel(const onnx::ModelProto &model, ExecutionMode mode);

  /**
   * Retrieve the set of non-constexpr nodes
   *
   * \param model The model to get the nodes from
   * \param mode The intended execution mode
   *
   * \return A vector of node names
   */
  static std::vector<std::string>
  listNonConstExprNodesModel(const onnx::ModelProto &model, ExecutionMode mode);

  ~Builder();

  /**
   * Add a new input tensor to the model
   *
   * \param tensorInfo The shape and type of the input tensor
   * \return The unique name of the input tensor
   */
  TensorId addInputTensor(const TensorInfo &tensorInfo);

  /**
   * Add a new preinitialized input tensor to the model
   *
   * \param initData The initial data of the input tensor
   * \return The unique name of the input tensor
   */
  TensorId addInitializedInputTensor(const ConstVoidData &initData);

  /**
   * Adds one of the outputs from a node in the graph into the list of output
   * tensors.
   */
  void addOutputTensor(const TensorId &arg0);

  /**
   * Return the builder interface for ai.onnx opset 6
   */
  AiOnnxOpset6 aiOnnxOpset6() { return AiOnnxOpset6(this->impl_); }

  /**
   * Return the builder interface for ai.onnx opset 7
   */
  AiOnnxOpset7 aiOnnxOpset7() { return AiOnnxOpset7(this->impl_); }

  /**
   * Return the builder interface for ai.onnx opset 7
   */
  AiOnnxOpset8 aiOnnxOpset8() { return AiOnnxOpset8(this->impl_); }

  /**
   * Return the builder interface for ai.onnx opset 9
   */
  AiOnnxOpset9 aiOnnxOpset9() { return AiOnnxOpset9(this->impl_); }

  /**
   * Return the builder interface for ai.onnx.ml opset 1
   */
  AiOnnxMlOpset1 aiOnnxMlOpset1() { return AiOnnxMlOpset1(this->impl_); }

  /**
   * Return the builder interface for ai.graphcore opset 1
   */
  AiGraphcoreOpset1 aiGraphcoreOpset1() {
    return AiGraphcoreOpset1(this->impl_);
  }

  // Add a custom op to the model
  // TODO : Think of a better name
  std::vector<TensorId>
  customOp(const OperatorIdentifier &opid,
           int opsetVersion,
           const std::vector<TensorId> &inputs,
           const unsigned numOutputs,
           const std::map<std::string, boost::any> &attributes,
           const std::string &name = "");

  /**
   * This is a helper function that will add a constant and a reshape using the
   * provided domain.
   */
  template <class T>
  TensorId reshape_const(T &t,
                         const std::vector<TensorId> &args,
                         const std::vector<int64_t> &shape,
                         const std::string &name = {}) {
    Shape s = {static_cast<int64_t>(shape.size())};
    TensorInfo tensorInfo("INT64", s);
    auto newShape = t.constant({shape.data(), tensorInfo}, name + "_const");
    return t.reshape({args[0], newShape}, name);
  }

  /**
   * Enable/disable recomputation of the output of the node in the backward
   * pass.
   *
   * \param nodeOutputName Name of the output tensor of the ONNX node
   * \param value If the recompute is enabled/disabled
   */
  void recomputeOutputInBackwardPass(const TensorId &nodeOutputName,
                                     bool value = true) {
    addNodeAttribute(sRecomputeOutputAttribute, value, {nodeOutputName});
  }

  /**
   * Enable/disable recomputation of the output of the node in the backward
   * pass.
   *
   * \param nodeOutputNames Names of the output tensors of the ONNX node
   * \param value If the recompute is enabled/disabled
   */
  void recomputeOutputInBackwardPass(const std::set<TensorId> &nodeOutputNames,
                                     bool value = true) {
    addNodeAttribute(sRecomputeOutputAttribute, value, nodeOutputNames);
  }

  /**
   * Set the virtual graph that computes the given node.  Applies when creating
   * a graph for a multi-IPU configuration.
   *
   * \param nodeOutputName Name of the output tensor of the ONNX node
   * \param value If the recompute is enabled/disabled
   */
  void virtualGraph(const TensorId &nodeOutputName, int64_t value = 0) {
    addNodeAttribute(sVirtualGraphAttribute, value, {nodeOutputName});
  }

  void setInplacePreferences(const TensorId &nodeOutputName,
                             const std::map<OpType, float> &prefs) {

    std::vector<OpType> names;
    std::vector<float> priorities;
    for (auto &x : prefs) {
      names.push_back(x.first);
      priorities.push_back(x.second);
    }
    addNodeAttribute(sInplaceOpNames, names, {nodeOutputName});
    addNodeAttribute(sInplaceOpPriorities, priorities, {nodeOutputName});
  }

  /**
   * Set an attribute that will be set on all subsequent operations
   */
  void setAttribute(const std::string &attribute, boost::any value);

  /**
   * Unset an attribute that will be set on all subsequent operations
   */
  void clearAttribute(const std::string &attribute);

  /**
   * Set the virtual graph that computes the given node.  Applies when creating
   * a graph for a multi-IPU configuration.
   *
   * \param nodeOutputNames Names of the output tensors of the ONNX node
   * \param value If the recompute is enabled/disabled
   */
  void virtualGraph(const std::set<TensorId> &nodeOutputNames,
                    int64_t value = 0) {
    addNodeAttribute(sVirtualGraphAttribute, value, nodeOutputNames);
  }

  /**
   * Add an attribute to the ONNX node which is uniquely identified by the
   * outputs.
   * This functions will throw an exception if it can't find the unique
   * node or the attribute already exists.
   *
   * \param attributeName The name of the attribute to add.
   * \param attributeValue An int64_t value of the attribute to add.
   * \param nodeOutputNames Names of the output tensors of the ONNX node used to
   *                        find the node in the ONNX model.
   */
  void addNodeAttribute(const std::string &attributeName,
                        const int64_t &attributeValue,
                        const std::set<TensorId> &nodeOutputNames);

  /**
   * Add an attribute to the ONNX node which is uniquely identified by the
   * outputs.
   * This functions will throw an exception if it can't find the unique
   * node or the attribute already exists.
   *
   * \param attributeName The name of the attribute to add.
   * \param attributeValue An std::vector<int64_t> value of the attribute to
   *                       add.
   * \param nodeOutputNames Names of the output tensors of the ONNX node used to
   *                        find the node in the ONNX model.
   */
  void addNodeAttribute(const std::string &attributeName,
                        const std::vector<int64_t> &attributeValue,
                        const std::set<TensorId> &nodeOutputNames);

  /**
   * Add an attribute to the ONNX node which is uniquely identified by the
   * outputs.
   * This functions will throw an exception if it can't find the unique
   * node or the attribute already exists.
   *
   * \param attributeName The name of the attribute to add.
   * \param attributeValue A float value of the attribute to add.
   * \param nodeOutputNames Names of the output tensors of the ONNX node used to
   *                        find the node in the ONNX model.
   */
  void addNodeAttribute(const std::string &attributeName,
                        const float &attributeValue,
                        const std::set<TensorId> &nodeOutputNames);

  /**
   * Add an attribute to the ONNX node which is uniquely identified by the
   * outputs.
   * This functions will throw an exception if it can't find the unique
   * node or the attribute already exists.
   *
   * \param attributeName The name of the attribute to add.
   * \param attributeValue An std::vector<float> value of the attribute to add.
   * \param nodeOutputNames Names of the output tensors of the ONNX node used to
   *                        find the node in the ONNX model.
   */
  void addNodeAttribute(const std::string &attributeName,
                        const std::vector<float> &attributeValue,
                        const std::set<TensorId> &nodeOutputNames);

  /**
   * Add an attribute to the ONNX node which is uniquely identified by the
   * outputs.
   * This functions will throw an exception if it can't find the unique
   * node or the attribute already exists.
   *
   * \param attributeName The name of the attribute to add.
   * \param attributeValue A std::string value of the attribute to add.
   * \param nodeOutputNames Names of the output tensors of the ONNX node used to
   *                        find the node in the ONNX model.
   */
  void addNodeAttribute(const std::string &attributeName,
                        const std::string &attributeValue,
                        const std::set<TensorId> &nodeOutputNames);

  void addNodeAttribute(const std::string &attributeName,
                        const char *attributeValue,
                        const std::set<TensorId> &nodeOutputNames);

  /**
   * Add an attribute to the ONNX node which is uniquely identified by the
   * outputs.
   * This functions will throw an exception if it can't find the unique
   * node or the attribute already exists.
   *
   * \param attributeName The name of the attribute to add.
   * \param attributeValue An std::vector<std::string> value of the attribute to
   *                       add.
   * \param nodeOutputNames Names of the output tensors of the ONNX node used to
   *                        find the node in the ONNX model.
   */
  void addNodeAttribute(const std::string &attributeName,
                        const std::vector<std::string> &attributeValue,
                        const std::set<TensorId> &nodeOutputNames);

  /**
   * Add an attribute to the ONNX node which is uniquely identified by the
   * outputs.
   * This functions will throw an exception if it can't find the unique
   * node or the attribute already exists.
   *
   * \param attributeName The name of the attribute to add.
   * \param attributeValue An bool value of the attribute to add
   * \param nodeOutputNames Names of the output tensors of the ONNX node used to
   *                        find the node in the ONNX model.
   */
  void addNodeAttribute(const std::string &attributeName,
                        const bool attributeValue,
                        const std::set<TensorId> &nodeOutputNames);

  /**
   * Add an attribute to the ONNX node which is uniquely identified by the
   * outputs.
   * This functions will throw an exception if it can't find the unique
   * node or the attribute already exists.
   *
   * \param attributeName The name of the attribute to add.
   * \param attributeValue An constant tensor initializer
   * \param nodeOutputNames Names of the output tensors of the ONNX node used to
   *                        find the node in the ONNX model.
   */
  void addNodeAttribute(const std::string &attributeName,
                        const ConstVoidData &attributeValue,
                        const std::set<TensorId> &nodeOutputNames);

  /**
   * Check whether the ONNX node has an attribute set.
   * This functions will throw an exception if it can't find the unique
   * node.
   *
   * \param attributeName The name of the attribute to find.
   * \param nodeOutputNames Names of the output tensors of the ONNX node used to
   *                        find the node in the ONNX model.
   */
  bool nodeHasAttribute(const std::string &attributeName,
                        const std::set<TensorId> &nodeOutputNames);

  /**
   * Get the int64_t value of the attribute for the ONNX node.
   * This functions will throw an exception if it can't find the unique
   * node or the attribute does not exist or it has not been set to the
   * int64_t type.
   *
   * \param attributeName The name of the attribute to find.
   * \param nodeOutputNames Names of the output tensors of the ONNX node used to
   *                        find the node in the ONNX model.
   * \return Value of the attribute
   */
  int64_t getInt64NodeAttribute(const std::string &attributeName,
                                const std::set<TensorId> &nodeOutputNames);

  /**
   * Get the std::vector<int64_t> value of the attribute for the ONNX node.
   * This functions will throw an exception if it can't find the unique
   * node or the attribute does not exist or it has not been set to the
   * std::vector<int64_t> type.
   *
   * \param attributeName The name of the attribute to find.
   * \param nodeOutputNames Names of the output tensors of the ONNX node used to
   *                        find the node in the ONNX model.
   * \return Value of the attribute
   */
  std::vector<int64_t>
  getInt64VectorNodeAttribute(const std::string &attributeName,
                              const std::set<TensorId> &nodeOutputNames);

  /**
   * Get the float value of the attribute for the ONNX node.
   * This functions will throw an exception if it can't find the unique
   * node or the attribute does not exist or it has not been set to the
   * float type.
   *
   * \param attributeName The name of the attribute to find.
   * \param nodeOutputNames Names of the output tensors of the ONNX node used to
   *                        find the node in the ONNX model.
   * \return Value of the attribute
   */
  float getFloatNodeAttribute(const std::string &attributeName,
                              const std::set<TensorId> &nodeOutputNames);

  /**
   * Get the std::vector<float> value of the attribute for the ONNX node.
   * This functions will throw an exception if it can't find the unique
   * node or the attribute does not exist.
   *
   * \param attributeName The name of the attribute to find.
   * \param nodeOutputNames Names of the output tensors of the ONNX node used to
   *                        find the node in the ONNX model.
   * \return Value of the attribute
   */
  std::vector<float>
  getFloatVectorNodeAttribute(const std::string &attributeName,
                              const std::set<TensorId> &nodeOutputNames);

  /**
   * Get the std::string value of the attribute for the ONNX node.
   * This functions will throw an exception if it can't find the unique
   * node or the attribute does not exist or it has not been set to the
   * std::string type.
   *
   * \param attributeName The name of the attribute to find.
   * \param nodeOutputNames Names of the output tensors of the ONNX node used to
   *                        find the node in the ONNX model.
   * \return Value of the attribute
   */
  std::string getStringNodeAttribute(const std::string &attributeName,
                                     const std::set<TensorId> &nodeOutputNames);

  /**
   * Get the std::vector<std::string> value of the attribute for the ONNX node.
   * This functions will throw an exception if it can't find the unique
   * node or the attribute does not exist.
   *
   * \param attributeName The name of the attribute to find.
   * \param nodeOutputNames Names of the output tensors of the ONNX node used to
   *                        find the node in the ONNX model.
   * \return Value of the attribute
   */
  std::vector<std::string>
  getStringVectorNodeAttribute(const std::string &attributeName,
                               const std::set<TensorId> &nodeOutputNames);

  bool getBoolNodeAttribute(const std::string &attributeName,
                            const std::set<TensorId> &nodeOutputNames);

  /**
   * Remove an attribute from the ONNX node.
   * This functions will throw an exception if it can't find the unique
   * node or the attribute does not exist.
   *
   * \param attributeName The name of the attribute to find.
   * \param nodeOutputNames Names of the output tensors of the ONNX node used to
   *                        find the node in the ONNX model.
   */
  void removeNodeAttribute(const std::string &attributeName,
                           const std::set<TensorId> &nodeOutputNames);

  /**
   * Get all the attribute names from the ONNX node.
   * This functions will throw an exception if it can't find the unique
   * node.
   *
   * \param nodeOutputNames Names of the output tensors of the ONNX node used to
   *                        find the node in the ONNX model.
   */
  std::vector<std::string>
  getAllNodeAttributeNames(const std::set<TensorId> &nodeOutputNames);

  /**
   * Get whether the given node will have its output recomputed in the backward
   * pass.
   *
   * \param nodeOutputName Name of the output tensor of the ONNX node used to
   *                        find the node in the ONNX model.
   */
  bool getRecomputeOutputInBackwardPass(const TensorId &nodeOutputName) {
    return getBoolNodeAttribute(sRecomputeOutputAttribute, {nodeOutputName});
  }

  /**
   * Get whether the given node will have its output recomputed in the backward
   * pass.
   *
   * \param nodeOutputNames Names of the output tensors of the ONNX node used to
   *                        find the node in the ONNX model.
   */
  bool
  getRecomputeOutputInBackwardPass(const std::set<TensorId> &nodeOutputNames) {
    return getBoolNodeAttribute(sRecomputeOutputAttribute, nodeOutputNames);
  }

  /**
   * Get the index of the virtual graph that computes this node. This applies
   * in a multi IPU system.
   *
   * \param nodeOutputName Name of the output tensor of the ONNX node used to
   *                       find the node in the ONNX model.
   */
  int64_t getVirtualGraph(const TensorId &nodeOutputName) {
    return getInt64NodeAttribute(sVirtualGraphAttribute, {nodeOutputName});
  }

  /**
   * Get the index of the virtual graph that computes this node. This applies
   * in a multi IPU system.
   *
   * \param nodeOutputNames Names of the output tensors of the ONNX node used to
   *                        find the node in the ONNX model.
   */
  int64_t getVirtualGraph(const std::set<TensorId> &nodeOutputNames) {
    return getInt64NodeAttribute(sVirtualGraphAttribute, nodeOutputNames);
  }

  /**
   * Retrieve the ONNX serialized ModelProto
   *
   * \return A serialized ONNX ModelProto
   */
  std::string getModelProto() const;

  /**
   * Retrieve the set of constexpr nodes
   *
   * \param mode The intended execution mode
   *
   * \return A vector of node names
   */
  std::vector<std::string> listConstExprNodes(ExecutionMode mode) const;

  /**
   * Retrieve the set of non-constexpr nodes
   *
   * \param mode The intended execution mode
   *
   * \return A vector of node names
   */
  std::vector<std::string> listNonConstExprNodes(ExecutionMode mode) const;

  /**
   * Return a list of ONNX graph input tensor ids
   *
   * \return A vector of input tensor names
   */
  std::vector<TensorId> getInputTensorIds() const;

  /**
   * Return a list of ONNX graph output tensor ids
   *
   * \return A vector of output tensor names
   */
  std::vector<TensorId> getOutputTensorIds() const;

  /**
   * Return a list of ONNX graph value tensor ids
   *
   * These tensors are stored in the `value_info` section
   * of the ONNX GraphProto structure.
   *
   * \return A vector of output tensor names
   */
  std::vector<TensorId> getValueTensorIds() const;

  /**
   * Return an ONNX graph tensor shape, from either the input,
   * output, or value_info lists in the GraphProto
   *
   * \param id Tensor id
   * \return A vector of tensor dimensions
   */
  std::vector<int64_t> getTensorShape(const TensorId id);

  /**
   * Push a name onto the name scope stack.
   *
   * The names of tensors and nodes added to the ONNX graph will be prefixed
   * with a concatenation of the names in the name stack.
   */
  void pushNameScope(const std::string &name);

  /**
   * Remove the last entry in the name scope stack
   */
  void popNameScope();

private:
  void configure();
  void configure(const std::string &modelProtoOrFilename);

  /**
   * Load a serialized ONNX ModelProto into the builder and validate it.
   *
   * \param modelProtoOrFilename Either an ONNX model protobuf, or the name of a
   *                             file containing an ONNX model protobuf.
   */
  void loadModelProto(const std::string &modelProtoOrFilename);

  std::unique_ptr<BuilderImpl> impl_;

  void verifyWindowParameters(TensorId input,
                              const std::vector<int64_t> strides,
                              const std::vector<int64_t> padding,
                              const std::vector<int64_t> dilation = {});
};

} // namespace poponnx
#endif // GUARD_BUILDER_H
