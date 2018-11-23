#ifndef GUARD_BUILDER_H
#define GUARD_BUILDER_H

#include <memory>
#include <string>
#include <vector>

#include <poponnx/names.hpp>

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
   * Add the absolute value operator to the model
   *
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#Abs
   *
   * \param args The tensor argument
   * \return The name of the result tensor
   */
  TensorId abs(const std::vector<TensorId> &args);

  /**
   * Add the arc-cosine operator to the model
   *
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#ACos
   *
   * \param args The tensor argument
   * \return The name of the result tensor
   */
  TensorId acos(const std::vector<TensorId> &args);

  /**
   * Add the arc-hyperbolic-cosine operator to the model
   *
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#Acosh
   *
   * \param args The tensor argument
   * \return The name of the result tensor
   */
  TensorId acosh(const std::vector<TensorId> &args);

  /**
   * Add the addition operator to the model
   *
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#Add
   *
   * \param args The tensor arguments
   * \return The name of the result tensor
   */
  TensorId add(const std::vector<TensorId> &args);

  /**
   * Add the logical AND operator to the model
   *
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#And
   *
   * \param args The tensor arguments
   * \return The name of the result tensor
   */
  TensorId logical_and(const std::vector<TensorId> &args);

  /**
   * Add the arc sine operator to the model
   *
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#Asin
   *
   * \param args The tensor argument
   * \return The name of the result tensor
   */
  TensorId asin(const std::vector<TensorId> &args);

  /**
   * Add the arc hyperbolic sine operator to the model
   *
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#Asinh
   *
   * \param args The tensor argument
   * \return The name of the result tensor
   */
  TensorId asinh(const std::vector<TensorId> &args);

  /**
   * Add the arc tangent operator to the model
   *
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#Atan
   *
   * \param args The tensor argument
   * \return The name of the result tensor
   */
  TensorId atan(const std::vector<TensorId> &args);

  /**
   * Add the arc hyperbolic tangent operator to the model
   *
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#Atanh
   *
   * \param args The tensor argument
   * \return The name of the result tensor
   */
  TensorId atanh(const std::vector<TensorId> &args);

  /**
   * Add the cast operator to the model
   *
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#Cast
   *
   * \param args The tensor argument
   * \return The name of the result tensor
   */
  TensorId cast(const std::vector<TensorId> &args);

  /**
   * Add the ceil operator to the model
   *
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#Ceil
   *
   * \param args The tensor argument
   * \return The name of the result tensor
   */
  TensorId ceil(const std::vector<TensorId> &args);

  /**
   * Add the cosine operator to the model
   *
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#Cos
   *
   * \param args The tensor argument
   * \return The name of the result tensor
   */
  TensorId cos(const std::vector<TensorId> &args);

  /**
   * Add the hyperbolic cosine operator to the model
   *
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#Cosh
   *
   * \param args The tensor argument
   * \return The name of the result tensor
   */
  TensorId cosh(const std::vector<TensorId> &args);

  /**
   * Add the divide operator to the model
   *
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#Div
   *
   * \param args The tensor arguments
   * \return The name of the result tensor
   */
  TensorId div(const std::vector<TensorId> &args);

  /**
   * Add the exponential linear operator to the model
   *
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#Elu
   *
   * \param args The tensor argument
   * \return The name of the result tensor
   */
  TensorId elu(const std::vector<TensorId> &args);

  /**
   * Add the equal-to operator to the model
   *
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#Equal
   *
   * \param args The tensor arguments
   * \return The name of the result tensor
   */
  TensorId equal(const std::vector<TensorId> &args);

  /**
   * Add the exponent operator to the model
   *
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#Exp
   *
   * \param args The tensor arguments
   * \return The name of the result tensor
   */
  TensorId exp(const std::vector<TensorId> &args);

  /**
   * Add the floor operator to the model
   *
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#Floor
   *
   * \param args The tensor argument
   * \return The name of the result tensor
   */
  TensorId floor(const std::vector<TensorId> &args);

  /**
   * Add the greater-than operator to the model
   *
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#Greater
   *
   * \param args The tensor arguments
   * \return The name of the result tensor
   */
  TensorId greater(const std::vector<TensorId> &args);

  /**
   * Add the identity operator to the model
   *
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#Identity
   *
   * \param args The tensor argument
   * \return The name of the result tensor
   */
  TensorId identity(const std::vector<TensorId> &args);

  /**
   * Add the less-than operator to the model
   *
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#Less
   *
   * \param args The tensor arguments
   * \return The name of the result tensor
   */
  TensorId less(const std::vector<TensorId> &args);

  /**
   * Add the logarithm operator to the model
   *
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#Log
   *
   * \param args The tensor arguments
   * \return The name of the result tensor
   */
  TensorId log(const std::vector<TensorId> &args);

  /**
   * Add the maximum operator to the model
   *
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#Max
   *
   * \param args The tensor arguments
   * \return The name of the result tensor
   */
  TensorId max(const std::vector<TensorId> &args);

  /**
   * Add the mean operator to the model
   *
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#Mean
   *
   * \param args The tensor arguments
   * \return The name of the result tensor
   */
  TensorId mean(const std::vector<TensorId> &args);

  /**
   * Add the minimum operator to the model
   *
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#Min
   *
   * \param args The tensor arguments
   * \return The name of the result tensor
   */
  TensorId min(const std::vector<TensorId> &args);

  /**
   * Add the multiply operator to the model
   *
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#Mul
   *
   * \param args The tensor arguments
   * \return The name of the result tensor
   */
  TensorId mul(const std::vector<TensorId> &args);

  /**
   * Add the negate operator to the model
   *
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#Neg
   *
   * \param args The tensor argument
   * \return The name of the result tensor
   */
  TensorId neg(const std::vector<TensorId> &args);

  /**
   * Add the logical NOT operator to the model
   *
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#Not
   *
   * \param args The tensor argument
   * \return The name of the result tensor
   */
  TensorId logical_not(const std::vector<TensorId> &args);

  /**
   * Add the logical OR operator to the model
   *
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#Or
   *
   * \param args The tensor arguments
   * \return The name of the result tensor
   */
  TensorId logical_or(const std::vector<TensorId> &args);

  /**
   * Add the power operator to the model
   *
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#Pow
   *
   * \param args The tensor arguments
   * \return The name of the result tensor
   */
  TensorId pow(const std::vector<TensorId> &args);

  /**
   * Add the reciprocal operator to the model
   *
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#Reciprocal
   *
   * \param args The tensor argument
   * \return The name of the result tensor
   */
  TensorId reciprocal(const std::vector<TensorId> &args);

  /**
   * Add the rectified linear operator to the model
   *
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#Relu
   *
   * \param args The tensor argument
   * \return The name of the result tensor
   */
  TensorId relu(const std::vector<TensorId> &args);

  /**
   * Add the sigmoid operator to the model
   *
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#Sigmoid
   *
   * \param args The tensor argument
   * \return The name of the result tensor
   */
  TensorId sigmoid(const std::vector<TensorId> &args);

  /**
   * Add the sine operator to the model
   *
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#Sin
   *
   * \param args The tensor argument
   * \return The name of the result tensor
   */
  TensorId sin(const std::vector<TensorId> &args);

  /**
   * Add the hyperbolic sine operator to the model
   *
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#Sinh
   *
   * \param args The tensor argument
   * \return The name of the result tensor
   */
  TensorId sinh(const std::vector<TensorId> &args);

  /**
   * Add the softsign operator to the model
   *
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#Softsign
   *
   * \param args The tensor argument
   * \return The name of the result tensor
   */
  TensorId softsign(const std::vector<TensorId> &args);

  /**
   * Add the square root operator to the model
   *
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#Sqrt
   *
   * \param args The tensor argument
   * \return The name of the result tensor
   */
  TensorId sqrt(const std::vector<TensorId> &args);

  /**
   * Add the subtract operator to the model
   *
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#Sub
   *
   * \param args The tensor arguments
   * \return The name of the result tensor
   */
  TensorId sub(const std::vector<TensorId> &args);

  /**
   * Add the variadic summation operator to the model
   *
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#Sum
   *
   * \param args The tensor arguments
   * \return The name of the result tensor
   */
  TensorId sum(const std::vector<TensorId> &args);

  /**
   * Add the tangent operator to the model
   *
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#Tan
   *
   * \param args The tensor argument
   * \return The name of the result tensor
   */
  TensorId tan(const std::vector<TensorId> &args);

  /**
   * Add the hyperbolic tangent operator to the model
   *
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#Tanh
   *
   * \param args The tensor argument
   * \return The name of the result tensor
   */
  TensorId tanh(const std::vector<TensorId> &args);

  /**
   * Add the logical XOR operator to the model
   *
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#Xor
   *
   * \param args The tensor argument
   * \return The name of the result tensor
   */
  TensorId logical_xor(const std::vector<TensorId> &args);

  /**
   * Add a convolution to the model
   *
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#Conv
   *
   * If the argument tensor can contain two tensors, they are the input and
   * kernel.  If it contains three tensors then they are the input, kernel and
   * bias tensors.
   *
   * \param args The data input, and the filter kernel, and optionally the bias
   * \param strides The filter stride in each of the spatial dimensions
   * \param padding The input padding in each of the spatial directions. This
   *                has the same format as the ONNX node, where the values are
   *                grouped [d1_begin, d2_begin, ..., d1_end, d2_end, ...]
   * \param dilation The dilation value along each spatial axis of the filter
   * \param groups The number of filter groups
   * \return The name of the result tensor
   */
  TensorId convolution(const std::vector<TensorId> &args,
                       const std::vector<int> strides,
                       const std::vector<int> padding,
                       const std::vector<int> dilation,
                       int groups);

  /**
   * Add an averagepool to the model
   *
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#AveragePool
   *
   * \param args The data input
   * \param kernel_shape The size of the kernel along each axis
   * \param strides The filter stride in each of the spatial dimensions
   * \param padding The input padding in each of the spatial directions. This
   *                has the same format as the ONNX node, where the values are
   *                grouped [d1_begin, d2_begin, ..., d1_end, d2_end, ...]
   * \return The name of the result tensor
   */
  TensorId averagepool(const std::vector<TensorId> &args,
                       const std::vector<int> kernel_shape,
                       const std::vector<int> strides,
                       const std::vector<int> padding);

  /**
   * Add a maxpool to the model
   *
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#MaxPool
   *
   * \param args The data input
   * \param kernel_shape The size of the kernel along each axis
   * \param strides The filter stride in each of the spatial dimensions
   * \param padding The input padding in each of the spatial directions. This
   *                has the same format as the ONNX node, where the values are
   *                grouped [d1_begin, d2_begin, ..., d1_end, d2_end, ...]
   * \return The name of the result tensor
   */
  TensorId maxpool(const std::vector<TensorId> &args,
                   const std::vector<int> kernel_shape,
                   const std::vector<int> strides,
                   const std::vector<int> padding);

  /**
   * Add a GEMM operation to the model
   *
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#Gemm
   *
   * \param args Tensor A, B and C
   * \param alpha Scalar multiplier for the product of input tensors A * B
   * \param beta Scalar multiplier for input tensor C
   * \param transA Whether A should be transposed
   * \param transB Whether B should be transposed
   * \return The name of the result tensor
   */
  TensorId gemm(const std::vector<TensorId> &args,
                float alpha,
                float beta,
                int transA,
                int transB);

  /**
   * Add a MatMul operation to the model
   *
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#MatMul
   *
   * \param args N-dimensional matrix A, and N-dimensional matrix B
   * \return The name of the result tensor
   */
  TensorId matmul(const std::vector<TensorId> &args);

  /**
   * Retrieve the ONNX serialized ModelProto
   *
   * \return A serialized ONNX ModelProto
   */
  std::string getModelProto() const;

  /**
   * Getter for ONNX graph input tensors
   *
   * \return A vector of input tensor names
   */
  std::vector<TensorId> getInputTensorIds() const;

  /**
   * Getter for ONNX graph output tensors
   *
   * \return A vector of output tensor names
   */
  std::vector<TensorId> getOutputTensorIds() const;

  /**
   * Getter for ONNX graph tensor shape
   *
   * \param id Tensor id
   * \return A vector of tensor dimensions
   */
  std::vector<int64_t> getTensorShape(const TensorId id);

private:
  std::unique_ptr<BuilderImpl> impl_;
};

} // namespace willow
#endif // GUARD_BUILDER_H
