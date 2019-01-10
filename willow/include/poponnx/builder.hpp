#ifndef GUARD_BUILDER_H
#define GUARD_BUILDER_H

#include <memory>
#include <set>
#include <string>
#include <vector>

#include <poponnx/names.hpp>
#include <poponnx/opidentifier.hpp>

#include <boost/any.hpp>

namespace poponnx {

class BuilderImpl;
class TensorInfo;
enum class DataType;

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
   * Add a constant to the model
   *
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#Constant
   *
   * \param initData The initial data of the input tensor
   * \param name Optional identifer for operation
   * \return The unique name of the constant's output tensor
   */
  TensorId constant(const ConstVoidData &initData, const std::string &name);

  /**
   * Add the absolute value operator to the model
   *
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#Abs
   *
   * \param args The tensor argument
   * \param name Optional identifer for operation
   * \return The name of the result tensor
   */
  TensorId abs(const std::vector<TensorId> &args, const std::string &name = {});

  /**
   * Add the arc-cosine operator to the model
   *
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#ACos
   *
   * \param args The tensor argument
   * \param name Optional identifer for operation
   * \return The name of the result tensor
   */
  TensorId acos(const std::vector<TensorId> &args,
                const std::string &name = {});

  /**
   * Add the arc-hyperbolic-cosine operator to the model
   *
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#Acosh
   *
   * \param args The tensor argument
   * \param name Optional identifer for operation
   * \return The name of the result tensor
   */
  TensorId acosh(const std::vector<TensorId> &args,
                 const std::string &name = {});

  /**
   * Add the addition operator to the model
   *
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#Add
   *
   * \param args The tensor arguments
   * \param name Optional identifer for operation
   * \return The name of the result tensor
   */
  TensorId add(const std::vector<TensorId> &args, const std::string &name = {});

  /**
   * Add the logical AND operator to the model
   *
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#And
   *
   * \param args The tensor arguments
   * \param name Optional identifer for operation
   * \return The name of the result tensor
   */
  TensorId logical_and(const std::vector<TensorId> &args,
                       const std::string &name = {});

  /**
   * Add the arc sine operator to the model
   *
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#Asin
   *
   * \param args The tensor argument
   * \param name Optional identifer for operation
   * \return The name of the result tensor
   */
  TensorId asin(const std::vector<TensorId> &args,
                const std::string &name = {});

  /**
   * Add the arc hyperbolic sine operator to the model
   *
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#Asinh
   *
   * \param args The tensor argument
   * \param name Optional identifer for operation
   * \return The name of the result tensor
   */
  TensorId asinh(const std::vector<TensorId> &args,
                 const std::string &name = {});

  /**
   * Add the arc tangent operator to the model
   *
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#Atan
   *
   * \param args The tensor argument
   * \param name Optional identifer for operation
   * \return The name of the result tensor
   */
  TensorId atan(const std::vector<TensorId> &args,
                const std::string &name = {});

  /**
   * Add the arc hyperbolic tangent operator to the model
   *
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#Atanh
   *
   * \param args The tensor argument
   * \param name Optional identifer for operation
   * \return The name of the result tensor
   */
  TensorId atanh(const std::vector<TensorId> &args,
                 const std::string &name = {});

  /**
   * Add the cast operator to the model
   *
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#Cast
   *
   * \param args The tensor argument
   * \param to   DataType of the output tensor
   * \param name Optional identifer for operation
   * \return The name of the result tensor
   */
  TensorId cast(const std::vector<TensorId> &args,
                DataType to,
                const std::string &name = {});

  /**
   * Add the ceil operator to the model
   *
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#Ceil
   *
   * \param args The tensor argument
   * \param name Optional identifer for operation
   * \return The name of the result tensor
   */
  TensorId ceil(const std::vector<TensorId> &args,
                const std::string &name = {});

  /**
   * Concatenate several tensors. The tensors are concatenated along the
   * specified dimension.
   *
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#concat
   *
   *  \param args       The tensors
   *  \param dimension  The dimensions to concatenate on
   *  \param name Optional identifer for operation
   *
   *  \return The name of the result tensor
   */
  TensorId concat(const std::vector<TensorId> &args,
                  int64_t dimension,
                  const std::string &name = {});

  /**
   * Add the cosine operator to the model
   *
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#Cos
   *
   * \param args The tensor argument
   * \param name Optional identifer for operation
   * \return The name of the result tensor
   */
  TensorId cos(const std::vector<TensorId> &args, const std::string &name = {});

  /**
   * Add the hyperbolic cosine operator to the model
   *
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#Cosh
   *
   * \param args The tensor argument
   * \param name Optional identifer for operation
   * \return The name of the result tensor
   */
  TensorId cosh(const std::vector<TensorId> &args,
                const std::string &name = {});

  /**
   * Add the divide operator to the model
   *
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#Div
   *
   * \param args The tensor arguments
   * \param name Optional identifer for operation
   * \return The name of the result tensor
   */
  TensorId div(const std::vector<TensorId> &args, const std::string &name = {});

  /**
   * Add the exponential linear operator to the model
   *
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#Elu
   *
   * \param args The tensor argument
   * \param name Optional identifer for operation
   * \return The name of the result tensor
   */
  TensorId elu(const std::vector<TensorId> &args, const std::string &name = {});

  /**
   * Add the equal-to operator to the model
   *
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#Equal
   *
   * \param args The tensor arguments
   * \param name Optional identifer for operation
   * \return The name of the result tensor
   */
  TensorId equal(const std::vector<TensorId> &args,
                 const std::string &name = {});

  /**
   * Add the exponent operator to the model
   *
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#Exp
   *
   * \param args The tensor arguments
   * \param name Optional identifer for operation
   * \return The name of the result tensor
   */
  TensorId exp(const std::vector<TensorId> &args, const std::string &name = {});

  /**
   * Add the floor operator to the model
   *
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#Floor
   *
   * \param args The tensor argument
   * \param name Optional identifer for operation
   * \return The name of the result tensor
   */
  TensorId floor(const std::vector<TensorId> &args,
                 const std::string &name = {});

  /**
   * Add the gather operator to the model
   *
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#Gather
   *
   * \param args The tensor argument
   * \param name Optional identifer for operation
   * \return The name of the result tensor
   */
  TensorId gather(const std::vector<TensorId> &args,
                  int64_t axis            = 0,
                  const std::string &name = {});

  /**
   * Add the greater-than operator to the model
   *
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#Greater
   *
   * \param args The tensor arguments
   * \param name Optional identifer for operation
   * \return The name of the result tensor
   */
  TensorId greater(const std::vector<TensorId> &args,
                   const std::string &name = {});

  /**
   * Add the identity operator to the model
   *
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#Identity
   *
   * \param args The tensor argument
   * \param name Optional identifer for operation
   * \return The name of the result tensor
   */
  TensorId identity(const std::vector<TensorId> &args,
                    const std::string &name = {});

  /**
   * Add the less-than operator to the model
   *
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#Less
   *
   * \param args The tensor arguments
   * \param name Optional identifer for operation
   * \return The name of the result tensor
   */
  TensorId less(const std::vector<TensorId> &args,
                const std::string &name = {});

  /**
   * Add the logarithm operator to the model
   *
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#Log
   *
   * \param args The tensor arguments
   * \param name Optional identifer for operation
   * \return The name of the result tensor
   */
  TensorId log(const std::vector<TensorId> &args, const std::string &name = {});

  /**
   * Add the maximum operator to the model
   *
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#Max
   *
   * \param args The tensor arguments
   * \param name Optional identifer for operation
   * \return The name of the result tensor
   */
  TensorId max(const std::vector<TensorId> &args, const std::string &name = {});

  /**
   * Add the mean operator to the model
   *
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#Mean
   *
   * \param args The tensor arguments
   * \param name Optional identifer for operation
   * \return The name of the result tensor
   */
  TensorId mean(const std::vector<TensorId> &args,
                const std::string &name = {});

  /**
   * Add the minimum operator to the model
   *
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#Min
   *
   * \param args The tensor arguments
   * \param name Optional identifer for operation
   * \return The name of the result tensor
   */
  TensorId min(const std::vector<TensorId> &args, const std::string &name = {});

  /**
   * Add the multiply operator to the model
   *
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#Mul
   *
   * \param args The tensor arguments
   * \param name Optional identifer for operation
   * \return The name of the result tensor
   */
  TensorId mul(const std::vector<TensorId> &args, const std::string &name = {});

  /**
   * Add the negate operator to the model
   *
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#Neg
   *
   * \param args The tensor argument
   * \param name Optional identifer for operation
   * \return The name of the result tensor
   */
  TensorId neg(const std::vector<TensorId> &args, const std::string &name = {});

  /**
   * Add the logical NOT operator to the model
   *
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#Not
   *
   * \param args The tensor argument
   * \param name Optional identifer for operation
   * \return The name of the result tensor
   */
  TensorId logical_not(const std::vector<TensorId> &args,
                       const std::string &name = {});

  /**
   * Add the logical OR operator to the model
   *
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#Or
   *
   * \param args The tensor arguments
   * \param name Optional identifer for operation
   * \return The name of the result tensor
   */
  TensorId logical_or(const std::vector<TensorId> &args,
                      const std::string &name = {});

  /**
   * Add the power operator to the model
   *
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#Pow
   *
   * \param args The tensor arguments
   * \param name Optional identifer for operation
   * \return The name of the result tensor
   */
  TensorId pow(const std::vector<TensorId> &args, const std::string &name = {});

  /**
   * Add the reciprocal operator to the model
   *
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#Reciprocal
   *
   * \param args The tensor argument
   * \param name Optional identifer for operation
   * \return The name of the result tensor
   */
  TensorId reciprocal(const std::vector<TensorId> &args,
                      const std::string &name = {});

  /**
   * Add the rectified linear operator to the model
   *
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#Relu
   *
   * \param args The tensor argument
   * \param name Optional identifer for operation
   * \return The name of the result tensor
   */
  TensorId relu(const std::vector<TensorId> &args,
                const std::string &name = {});

  /**
   * Add the sigmoid operator to the model
   *
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#Sigmoid
   *
   * \param args The tensor argument
   * \param name Optional identifer for operation
   * \return The name of the result tensor
   */
  TensorId sigmoid(const std::vector<TensorId> &args,
                   const std::string &name = {});

  /**
   * Add the sine operator to the model
   *
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#Sin
   *
   * \param args The tensor argument
   * \param name Optional identifer for operation
   * \return The name of the result tensor
   */
  TensorId sin(const std::vector<TensorId> &args, const std::string &name = {});

  /**
   * Add the hyperbolic sine operator to the model
   *
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#Sinh
   *
   * \param args The tensor argument
   * \param name Optional identifer for operation
   * \return The name of the result tensor
   */
  TensorId sinh(const std::vector<TensorId> &args,
                const std::string &name = {});

  /**
   * Add the softsign operator to the model
   *
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#Softsign
   *
   * \param args The tensor argument
   * \param name Optional identifer for operation
   * \return The name of the result tensor
   */
  TensorId softsign(const std::vector<TensorId> &args,
                    const std::string &name = {});

  /**
   * Add the square root operator to the model
   *
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#Sqrt
   *
   * \param args The tensor argument
   * \param name Optional identifer for operation
   * \return The name of the result tensor
   */
  TensorId sqrt(const std::vector<TensorId> &args,
                const std::string &name = {});

  /**
   * Add the squeeze operator to the model
   *
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#Squeeze
   *
   * \param args The tensor argument
   * \param axes The dimensions to squeeze
   * \param name Optional identifer for operation
   * \return The name of the result tensor
   */
  TensorId squeeze(const std::vector<TensorId> &args,
                   const std::vector<int64_t> axes,
                   const std::string &name = {});

  /**
   * Add the unsqueeze operator to the model
   *
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#Unsqueeze
   *
   * \param args The tensor argument
   * \param axes The dimensions to insert
   * \param name Optional identifer for operation
   * \return The name of the result tensor
   */
  TensorId unsqueeze(const std::vector<TensorId> &args,
                     const std::vector<int64_t> axes,
                     const std::string &name = {});

  /**
   * Add the subtract operator to the model
   *
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#Sub
   *
   * \param args The tensor arguments
   * \param name Optional identifer for operation
   * \return The name of the result tensor
   */
  TensorId sub(const std::vector<TensorId> &args, const std::string &name = {});

  /**
   * Add the variadic summation operator to the model
   *
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#Sum
   *
   * \param args The tensor arguments
   * \param name Optional identifer for operation
   * \return The name of the result tensor
   */
  TensorId sum(const std::vector<TensorId> &args, const std::string &name = {});

  /**
   * Add the tangent operator to the model
   *
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#Tan
   *
   * \param args The tensor argument
   * \param name Optional identifer for operation
   * \return The name of the result tensor
   */
  TensorId tan(const std::vector<TensorId> &args, const std::string &name = {});

  /**
   * Add the hyperbolic tangent operator to the model
   *
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#Tanh
   *
   * \param args The tensor argument
   * \param name Optional identifer for operation
   * \return The name of the result tensor
   */
  TensorId tanh(const std::vector<TensorId> &args,
                const std::string &name = {});

  /**
   * Add the logical XOR operator to the model
   *
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#Xor
   *
   * \param args The tensor argument
   * \param name Optional identifer for operation
   * \return The name of the result tensor
   */
  TensorId logical_xor(const std::vector<TensorId> &args,
                       const std::string &name = {});

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
   * \param cacheOperation Whether to cache the part of the Poplar graph
   *                              for this convolution (and any corresponding
   *                              training pass convolutions). Caching of the
   *                              Poplar graph will reduce the size of the
   *                              program, but it will introduce extra copies
   *                              to the execution program.
   * \param name Optional identifer for operation
   * \return The name of the result tensor
   */
  TensorId convolution(const std::vector<TensorId> &args,
                       const std::vector<int64_t> strides,
                       const std::vector<int64_t> padding,
                       const std::vector<int64_t> dilation,
                       int64_t groups          = 1,
                       bool cacheOperation     = false,
                       const std::string &name = {});

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
   * \param name Optional identifer for operation
   * \return The name of the result tensor
   */
  TensorId averagepool(const std::vector<TensorId> &args,
                       const std::vector<int64_t> kernel_shape,
                       const std::vector<int64_t> strides,
                       const std::vector<int64_t> padding,
                       const std::string &name = {});

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
   * \param name Optional identifer for operation
   * \return The name of the result tensor
   */
  TensorId maxpool(const std::vector<TensorId> &args,
                   const std::vector<int64_t> kernel_shape,
                   const std::vector<int64_t> strides,
                   const std::vector<int64_t> padding,
                   const std::string &name = {});

  /**
   * Add a LSTM operation to the model
   *
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#LSTM
   *
   * \param args Tensor X, W, R, B, sequence_lens, initial_h and initial_c
   * \param name Optional identifer for operation
   * \return The name of the result tensor
   */
  std::tuple<TensorId, TensorId, TensorId>
  lstm(const std::vector<TensorId> &args, const std::string &name = {});

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
   * \param name Optional identifer for operation
   * \return The name of the result tensor
   */
  TensorId gemm(const std::vector<TensorId> &args,
                float alpha,
                float beta,
                int64_t transA,
                int64_t transB,
                const std::string &name = {});

  /**
   * Add a Pad operation to the model
   *
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#Pad
   *
   * \param args Tensor T
   * \param mode Three modes: constant, reflect, edge
   * \param pads The input padding in each of the spatial directions. This
   *                has the same format as the ONNX node, where the values are
   *                grouped [d1_begin, d2_begin, ..., d1_end, d2_end, ...]
   * \param value The value to be filled
   * \param name Optional identifer for operation
   */
  TensorId pad(const std::vector<TensorId> &args,
               std::string mode,
               const std::vector<int64_t> pads,
               float value,
               const std::string &name = {});

  /**
   * Add a MatMul operation to the model
   *
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#MatMul
   *
   * \param args N-dimensional matrix A, and N-dimensional matrix B
   * \param name Optional identifer for operation
   * \return The name of the result tensor
   */
  TensorId matmul(const std::vector<TensorId> &args,
                  const std::string &name = {});

  /**
   * Add a Slice operation to the model
   *
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#Slice
   *
   * \param args Tensor T
   * \param axes Axes that `starts` and `ends` apply to.
   * \param starts Starting indices of corresponding axis in `axes`
   * \param ends Ending indices (exclusive) of corresponding axis in `axes`
   * \return The name of the result tensor
   *
   */
  TensorId slice(const std::vector<TensorId> &args,
                 const std::vector<int64_t> &axes,
                 const std::vector<int64_t> &starts,
                 const std::vector<int64_t> &ends,
                 const std::string &name = {});

  /**
   * Add a Softmax operation to the model
   *
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#Softmax
   *
   * \param args Tensor T
   * \param name Optional identifer for operation
   * \return The name of the result tensor
   *
   */
  TensorId softmax(const std::vector<TensorId> &args,
                   const std::string &name = {});

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

  /**
   * Add a Transpose operation to the model
   *
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#Transpose
   *
   * \param args Tensor T
   * \param perm The new axes permutation
   * \param name Optional identifer for operation
   * \return The name of the result tensor
   *
   */
  TensorId transpose(const std::vector<TensorId> &args,
                     const std::vector<int64_t> &perm,
                     const std::string &name = {});

  /**
   * Add the reshape operator to the model
   *
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#Reshape
   *
   * \param args The tensor arguments
   * \param name Optional identifer for operation
   * \return The name of the output, reshaped tensor
   */
  TensorId reshape(const std::vector<TensorId> &args,
                   const std::string &name = {});

  /**
   * Add the reshape operator to the model
   *
   * The new shape is given as a parameter of the function
   *
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#Reshape
   *
   * \param args The tensor arguments
   * \param shape The new shape for the tensor
   * \param name Optional identifer for operation
   * \return The name of the output, reshaped tensor
   */
  TensorId reshape_const(const std::vector<TensorId> &args,
                         const std::vector<int64_t> &shape,
                         const std::string &name = {});

  // The set of output from the batchNormalization test
  // Would have liked to use std::optional instead of pointers
  struct BatchNormalizationTrainingOutputs {
    TensorId y;
    TensorId mean;
    TensorId var;
    TensorId savedMean;
    TensorId savedVar;
  };

  /**
   * Add a Batch Normalization operation to the model (for training models)
   *
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#batchnormalization
   *
   * \param x the input tensor
   * \param scale the scale tensor
   * \param b the bias tensor
   * \param mean the mean tenor
   * \param var the variance tensor
   * \param epsilon a small delta applied to avoid division by zero
   * \param momentum the fraction used when updating the mean/variance tensors
   * \param spatial see ONNX operation specificiation
   * \param name Optional identifer for operation
   * \return a struct containing all of the outputs of the batch-normalization
   */
  BatchNormalizationTrainingOutputs
  batchnormalizationTraining(const TensorId x,
                             const TensorId scale,
                             const TensorId b,
                             const TensorId mean,
                             const TensorId var,
                             const float epsilon     = 1e-5f,
                             const float momentum    = 0.9f,
                             const int spatial       = 1,
                             const std::string &name = {});

  /**
   * Add a Batch Normalization operation to the model (for non-training)
   *
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#batchnormalization
   *
   * \param x the input tensor
   * \param scale the scale tensor
   * \param b the bias tensor
   * \param mean the mean tensor
   * \param var the variance tensor
   * \param epsilon a small value added to prevent division by zero
   * \param momentum the update fraction for changing the mean/variance
   * \param spatial see ONNX operator specification
   * \param name Optional identifer for operation
   * \return The normalized input tensor
   */
  TensorId batchnormalizationTesting(const TensorId x,
                                     const TensorId scale,
                                     const TensorId b,
                                     const TensorId mean,
                                     const TensorId var,
                                     const float epsilon     = 1e-5f,
                                     const float momentum    = 0.9f,
                                     const int spatial       = 1,
                                     const std::string &name = {});

  // Add a custom op to the model
  // TODO : Think of a better name
  std::vector<TensorId>
  customOp(const OperatorIdentifier &opid,
           const std::vector<boost::any> &inputs,
           const unsigned numOutputs,
           const std::vector<std::pair<std::string, boost::any>> &attributes,
           const std::string &name = "");

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
   * When an ONNX model is loaded, all the tensor names are made unique. This
   * function returns the translation table from the original tensor names to
   * the unique names for all the tensors which were translated.
   *
   * \return Translation table
   */
  const std::map<std::string, TensorId> getTensorTranslation() const;

  /**
   * Retrieve the ONNX serialized ModelProto
   *
   * \return A serialized ONNX ModelProto
   */
  std::string getModelProto() const;

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
};

} // namespace poponnx
#endif // GUARD_BUILDER_H
