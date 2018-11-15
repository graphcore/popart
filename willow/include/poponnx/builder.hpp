#ifndef GUARD_BUILDER_H
#define GUARD_BUILDER_H

#include <memory>
#include <string>
#include <vector>

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
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#Add
   *
   * \param arg0 The name of the first argument tensor
   * \param arg1 The name of the second argument tensor
   * \return The name of the result tensor
   */
  std::string add(const std::string &arg0, const std::string &arg1);

  /**
   * Add a convolution to the model
   *
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#Conv
   *
   * \param arg0 The data input
   * \param arg1 The filter kernel
   * \param strides The filter stride in each of the spatial dimensions
   * \param padding The input padding in each of the spatial directions. This
   *                has the same format as the ONNX node, where the values are
   *                grouped [d1_begin, d2_begin, ..., d1_end, d2_end, ...]
   * \param dilation The dilation value along each spatial axis of the filter
   * \param groups The number of filter groups
   * \return The name of the result tensor
   */
  std::string convolution(const std::string &arg0,
                          const std::string &arg1,
                          const std::vector<int> strides,
                          const std::vector<int> padding,
                          const std::vector<int> dilation,
                          int groups);

  /**
   * Add a convolution with a bias to the model
   *
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#Conv
   *
   * \param arg0 The data input
   * \param arg1 The filter kernel
   * \param arg2 The bias tensor
   * \param strides The filter stride in each of the spatial dimensions
   * \param padding The input padding in each of the spatial directions. This
   *                has the same format as the ONNX node, where the values are
   *                grouped [d1_begin, d2_begin, ..., d1_end, d2_end, ...]
   * \param dilation The dilation value along each spatial axis of the filter
   * \param groups The number of filter groups
   * \return The name of the result tensor
   */
  std::string convolutionWithBias(const std::string &arg0,
                                  const std::string &arg1,
                                  const std::string &arg2,
                                  const std::vector<int> strides,
                                  const std::vector<int> padding,
                                  const std::vector<int> dilation,
                                  int groups);

  /**
   * Add a GEMM operation to the model
   *
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#Gemm
   *
   * \param arg0 Tensor A
   * \param arg1 Tensor B
   * \param arg2 Tensor C
   * \param alpha Scalar multiplier for the product of input tensors A * B
   * \param beta Scalar multiplier for input tensor C
   * \param transA Whether A should be transposed
   * \param transB Whether B should be transposed
   * \return The name of the result tensor
   */
  std::string gemm(const std::string &arg0,
                   const std::string &arg1,
                   const std::string &arg2,
                   float alpha,
                   float beta,
                   int transA,
                   int transB);

  /**
   * Add a MatMul operation to the model
   *
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#MatMul
   *
   * \param arg0 N-dimensional matrix A
   * \param arg1 N-dimensional matrix B
   * \return The name of the result tensor
   */
  std::string matmul(const std::string &arg0, const std::string &arg1);

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
