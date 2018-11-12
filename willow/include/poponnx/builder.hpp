#ifndef GUARD_BUILDER_H
#define GUARD_BUILDER_H

#include <memory>
#include <string>

namespace willow {

class BuilderImpl;
class TensorInfo;

/**
 * An interface for a Builder, used for creating ONNX graphs.
 */
class Builder {
public:
  Builder();
  ~Builder();

  /**
   * Add a new input tensor to the model
   *
   * \param tensorInfo The shape and type of the input tensor
   * \return The unique name of the input tensor
   */
  std::string addInputTensor(const TensorInfo &tensorInfo);

  /**
   * Adds one of the outputs from a node in the graph into the list of output
   * tensors.
   */
  void addOutputTensor(const std::string &arg0);

  /**
   * Add the Addition operator to the model
   *
   * \param arg0 The name of the first argument tensor
   * \param arg1 The name of the second argument tensor
   * \return The name of the result tensor
   */
  std::string add(const std::string &arg0, const std::string &arg1);

  /**
   * Retrieve the ONNX serialized ModelProto
   *
   * \return A serialized ONNX ModelProto
   */
  std::string getModelProto() const;

private:
  std::unique_ptr<BuilderImpl> impl_;
};

} // namespace willow
#endif // GUARD_BUILDER_H
