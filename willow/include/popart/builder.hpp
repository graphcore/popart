// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_BUILDER_HPP
#define GUARD_BUILDER_HPP

#include <memory>
#include <set>
#include <string>
#include <vector>

#include <popart/debugcontext.hpp>
#include <popart/names.hpp>
#include <popart/op.hpp>
#include <popart/op/loss.hpp>
#include <popart/opidentifier.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/tensorlocation.hpp>

#include <popart/vendored/any.hpp>
#include <popart/vendored/optional.hpp>

namespace popart {

class BuilderImpl;
/**
 * \class Builder
 * \brief A builder interface for creating ONNX graphs.
 *
 * ONNX defines a specification for describing graphs and serialising them as
 * protobuf files. This class provides a builder interface for creating such a
 * graph.
 *
 * Note, in ONNX, all Ops belong to an "Opset". The Builder itself does not have
 * methods for creating Ops in the ONNX graph, but instead has accessors to
 * Opsets, like AiGraphcoreOpset1, which contain the methods for creating Ops in
 * the graph.
 */
class Builder;
enum class DataType;
enum class RecomputeType;

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
  // Builds an op for specified bitwise operator id.
  TensorId bitwiseGenericOp(const OperatorIdentifier &opid,
                            const std::vector<TensorId> &args,
                            const DebugContext &debugContext = {});

protected:
  using DomainOpSet::impl;

  int getOpsetVersion() const override { return 1; }

public:
  AiGraphcoreOpset1(std::unique_ptr<BuilderImpl> &impl_) : DomainOpSet(impl_) {}

  /**
   * Add a group normalization operation to the model.
   *
   * This is a Poplar extension.
   *
   * The group will be created from a strided input.
   *
   * \param args A vector of input tensors: [x, scale, bias].
   * \param num_groups The number of groups to separate the channels into.
   * \param epsilon The epsilon value to use to avoid division by zero.
   * \param debugContext Optional debug context.
   * \return A vector of tensors: [y, mean, var].
   */
  std::vector<TensorId>
  groupnormalization(const std::vector<TensorId> &args,
                     int64_t num_groups,
                     float epsilon                    = 1e-05f,
                     const DebugContext &debugContext = {});

  /**
   * Add a multi-convolution to the model.
   *
   * Using this multi-convolution API ensures that the convolutions are
   * executed in parallel on the device.
   *
   * Functionally, a multi-convolution is equivalent to a series of single
   * convolutions. Using this multi-convolution API is always equivalent to
   * calling the single-convolution API (conv) once for each argument.
   *
   * For example, calling:
   *
   *     A0 = conv({X0, W0, B0})
   *     A1 = conv({X1, W1})
   *
   * Is functionally equivalent to calling:
   *
   *     {A0, A1} = multiconv({{X0, W0, B0}, {X1, Q1}).
   *
   * It is possible that any two convolutions cannot be executed in parallel
   * due to topological constraints. For example, the following:
   *
   *     B = conv({A, W0});
   *     C = B + A
   *     D = conv({C, W1});
   *
   * Cannot be converted to:
   *
   *     {B, D} = multiconv({{A, W0}, {C, W1}}).
   *
   * Note that it is not possible to create such a cycle by adding a
   * multi-convolution with this API.
   *
   * Calls to multiconv() are mapped to
   * poplar::poplin::multiconv::convolution().
   *
   *
   * \param tensors List of [DataId, WeightId, BiasId (optional)] for each
   *        convolution.
   * \param dilations The dilations attributes for each convolution.
   * \param inDilations The input dilations attributes for each convolution.
   * \param pads The pads for each convolution.
   * \param outPads The output padding for each convolution.
   * \param strides The strides for each convolution.
   * \param availableMemoryProportions The available memory proportions per
            conv, each [0, 1).
   * \param partialsTypes The partials type per convolution.
   * \param planType Run convolutions in parallel or series.
   * \param perConvReservedTiles Tiles to reserve per convolution when planning.
   * \param cycleBackOff Cycle back-off proportion, [0, 1).
   * \param debugContext Optional debug context.
   *
   * All input vectors must be either empty, or equal in length to
   * the number of convolutions. Note that groups for each convolution are
   * automatically inferred from the shapes of the data and weight inputs.
   *
   * \return The TensorId of the output tensor from each convolution.
   *
   */
  std::vector<TensorId>
  multiconv(const MultiConvInputs &tensors,
            const MultiConvDilations &dilations                  = {},
            const MultiConvDilations &inDilations                = {},
            const MultiConvPads &pads                            = {},
            const MultiConvPads &outPads                         = {},
            const MultiConvStrides &strides                      = {},
            const std::vector<float> &availableMemoryProportions = {},
            const std::vector<std::string> &partialsTypes        = {},
            const nonstd::optional<std::string> planType     = nonstd::nullopt,
            const nonstd::optional<int> perConvReservedTiles = nonstd::nullopt,
            const nonstd::optional<float> cycleBackOff       = nonstd::nullopt,
            const DebugContext &debugContext                 = {});

  /**
   * Add a sub-sample operation to the model.
   *
   * This is a Poplar extension.
   *
   * If multiple tensors are provided that strides will applied to them all.
   *
   * \param args Vector of tensor ids to sub-sample.
   * \param strides The strides to use.
   * \param debugContext Optional debug context.
   * \return The name of the result tensor.
   */
  TensorId subsample(const std::vector<TensorId> &args,
                     const std::vector<int64_t> &strides,
                     const DebugContext &debugContext = {});

  /**
   * Add a print tensor operation to the model.
   *
   * This is a Poplar extension.
   *
   * \param args Vector of tensor ids to print.
   * \param print_gradient
   * \param debugContext Optional debug context.
   * \param title
   * \return The name of the result tensor.
   */
  TensorId printtensor(const std::vector<TensorId> &args,
                       int64_t print_gradient           = 1,
                       const DebugContext &debugContext = {},
                       const std::string &title         = {});

  /**
   * Add a no-op operation to the model.
   *
   * \param args Vector of input tensor ids.
   * \param debugContext Optional debug context.
   * \return The name of the result tensor.
   */
  TensorId nop(const std::vector<TensorId> &args,
               const DebugContext &debugContext = {});

  /**
   * Add a scale operation to the model.
   *
   * This is a Poplar extension.
   *
   * \param args Vector of input tensor ids.
   * \param scale The scale to apply.
   * \param debugContext Optional debug context.
   * \return The name of the result tensor.
   */
  TensorId scale(const std::vector<TensorId> &args,
                 float scale,
                 const DebugContext &debugContext = {});
  /**
   * Add a scaled add operation to the model.
   *
   *     X = scale0 * T0 + scale1 * T1
   *
   * \param args Vector of input tensor ids: [T0, T1, scale0, scale1].
   * \param scale0 The scale to apply (if no \c scale0 tensor is supplied).
   * \param scale1 The scale to apply (if no \c scale1 tensor is supplied).
   * \param debugContext Optional debug context.
   * \return The name of the result tensor.
   */
  TensorId scaledadd(const std::vector<TensorId> &args,
                     float scale0,
                     float scale1,
                     const DebugContext &debugContext = {});

  std::vector<TensorId> lstm(const std::vector<TensorId> &args,
                             int64_t outputFullSequence,
                             const DebugContext &debugContext = {});
  /**
   * Add a GELU operation to the model.
   *
   * This is a Poplar extension.
   *
   * \param args Vector of input tensor ids.
   * \param debugContext Optional debug context.
   * \return The name of the result tensor.
   */
  TensorId gelu(const std::vector<TensorId> &args,
                const DebugContext &debugContext = {});

  /**
   * Add a detach operation to the model.
   *
   *
   * \param args Vector of input tensor ids.
   * \param debugContext Optional debug context.
   * \return The name of the result tensor.
   */
  TensorId detach(const std::vector<TensorId> &args,
                  const DebugContext &debugContext = {});

  /**
   * Add the \c DepthToSpace to the model.
   * (This allows DepthToSpace_11 to be targeted from earlier opsets.)
   *
   * The purpose of Depth to Space, also known as pixel shuffling, is to
   * rearrange data from the depth (channels) dimension into the spacial (width
   * and height) dimensions. It is an efficient means of learning upsampling
   * alongside mixing convolution with bilinear interpolation and using
   * transpose convolution.
   *
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#DepthToSpace
   *
   * \param args Vector containing single tensor input id.
   * \param blocksize Indicates the scale factor: if the input is [N, C, H, W]
   *     and the blocksize is B, the output will be [N, C/(B*B), H*B, W*B].
   * \param mode Specifies how the data is rearranged:
   *   *  "DCR": depth-column-row order
   *   *  "CRD": column-row-depth order
   * \param debugContext Optional debug context.
   * \return A tensor which is a rearrangement of the input tensor.
   */
  TensorId depthtospace(const std::vector<TensorId> &args,
                        int64_t blocksize,
                        const std::string &mode          = "DCR",
                        const DebugContext &debugContext = {});

  /**
   * Add a \c Round operation to the model.
   * (This allows \c Round_11 to be targeted from earlier opsets.)
   *
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#Round
   *
   * \param args Vector of input tensor ids.
   * \param debugContext Optional debug context.
   * \return The normalized output tensor ids.
   */
  TensorId round(const std::vector<TensorId> &args,
                 const DebugContext &debugContext = {});

  /**
   * Add an init operation to the model.
   *
   * \param shape Shape of the tensor to initialise.
   * \param data_type Data type to initialise tensor with.
   * \param init_type Mode of tensor initialisations.
   * \param batch_axis Axis relative to batch size.
   * \param debugContext Optional debug context.
   * \return The name of the result tensor.
   */
  TensorId init(Attributes::Ints shape,
                Attributes::Int data_type,
                Attributes::Int init_type,
                Attributes::Int batch_axis,
                const DebugContext &debugContext = {});

  /**
   * Add an init operation to the model.
   *
   * \param shape Shape of the tensor to initialise.
   * \param data_type Data type to initialise tensor with.
   * \param init_type Mode of tensor initialisations.
   * \param debugContext Optional debug context.
   * \return The name of the result tensor.
   */
  TensorId init(Attributes::Ints shape,
                Attributes::Int data_type,
                Attributes::Int init_type,
                const DebugContext &debugContext = {});

  /**
   * Add a dynamic slice operation to the model.
   *
   * Creates a new slice tensor.
   * For example:
   *
   *     slice = tensor[offset]
   *
   * \param args Vector of input tensor ids: [tensor, offset].
   * \param axes Axes along which to slice.
   * \param sizes Size of the slice in each axis.
   * \param debugContext Optional debug context.
   * \return The name of the result tensor.
   */
  TensorId dynamicslice(const std::vector<TensorId> &args,
                        Attributes::Ints axes,
                        Attributes::Ints sizes,
                        Attributes::Int noOverlap,
                        const DebugContext &debugContext = {});
  /**
   * Add a dynamic update operation to the model.
   *
   * Creates a copy of a \c tensor with a \c slice inserted at \c offset.
   * For example:
   *
   *     out = tensor, out[offset] = slice
   *
   * \param args Vector of input tensor ids: [tensor, offset, slice].
   * \param axes Axes along which to update.
   * \param sizes Size of the slice in each axis.
   * \param debugContext Optional debug context.
   * \return The name of the result tensor.
   */
  TensorId dynamicupdate(const std::vector<TensorId> &args,
                         Attributes::Ints axes,
                         Attributes::Ints sizes,
                         Attributes::Int noOverlap,
                         const DebugContext &debugContext = {});
  /**
   * Add a dynamic zero operation to the model.
   *
   * Creates a copy of \c tensor with a slice at \c offset set to zero.
   * For example:
   *
   *     out = tensor, out[offset] = 0.0
   *
   * \param args Vector of input tensor ids [tensor, offset].
   * \param axes Axes along which to erase.
   * \param sizes Size of the slice in each axis.
   * \param debugContext Optional debug context.
   * \return The name of the result tensor.
   */
  TensorId dynamiczero(const std::vector<TensorId> &args,
                       Attributes::Ints axes,
                       Attributes::Ints sizes,
                       const DebugContext &debugContext = {});
  /**
   * Add a dynamic add operation to the model.
   *
   * Creates a copy of \c tensor with \c slice added at \c offset.
   * For example:
   *
   *     out = tensor, out[offset] += slice
   *
   * \param args Vector of input tensor ids: [tensor, offset, slice].
   * \param axes Axes along which to add.
   * \param sizes Size of the slice in each axis.
   * \param debugContext Optional debug context.
   * \return The name of the result tensor.
   */
  TensorId dynamicadd(const std::vector<TensorId> &args,
                      Attributes::Ints axes,
                      Attributes::Ints sizes,
                      const DebugContext &debugContext = {});

  /**
   * Slice a 2D tensor based on offsets specified by a tensor.
   *
   * The outermost dimension is sliced;
   *   tOut[tOutOffset:tOutOffset+tN][...] = tIn[tInOffset:tInOffset+tN][...]
   * for each entry in tN/tInOffset/tOutOffset; entries after the first tN==0
   * may be ignored. Unreferenced elements of tOut are zeroed if zeroUnused is
   * set. The same output element should not be written by multiple inputs.
   *
   * tIn and tOut must have rank greater than or equal to 2. The outer dimension
   * is sliced; the product of the inner dimensions must match. tInOffset,
   * tOutOffset and tN must be 1d and the same size. \param [source,
   * destination, N, sourceOffset, destinationOffset] \param zeroUnused Whether
   * to zero unreferenced tOut elements. \param debugContext     Optional debug
   * context.
   */
  TensorId sequenceslice(const std::vector<TensorId> &args,
                         Attributes::Int zeroUnused,
                         const DebugContext &debugContext = {});

  /**
   * Add a call operation to the model
   *
   * This is a Poplar extension, to expose manual code re-use to
   * the builder.
   *
   * \param args Vector of input tensor ids.
   * \param callee The subgraph to call into.
   * \param debugContext Optional debug context.
   * \return A vector of tensors; the subgraph outputs.
   */
  std::vector<TensorId> call(const std::vector<TensorId> &args,
                             unsigned num_outputs,
                             const Builder &callee,
                             const DebugContext &debugContext = {});

  /**
   * Add a replicated all-reduce operation to the model.
   *
   * This is a Poplar extension, to expose manual code re-use to
   * the builder.
   *
   * \param args Vector of input tensor ids to reduce across.
   * \param debugContext Optional debug context.
   * \return The name of the result tensor.
   */
  TensorId replicatedallreduce(const std::vector<TensorId> &args,
                               const DebugContext &debugContext = {});

  /**
   * Add an \c l1 loss operation to the model.
   *
   * Calculates the mean absolute error between each element in the input with
   * a zero target.
   *
   * \param args Vector of input tensor ids.
   * \param lambda Scale factor of L1 loss.
   * \param reduction Type of reduction to perform on the individual losses.
   * \param debugContext Optional debug context.
   * \return The name of the result tensor.
   */
  TensorId l1loss(const std::vector<TensorId> &args,
                  const float lambda,
                  const ReductionType reduction    = ReductionType::Mean,
                  const DebugContext &debugContext = {});

  /**
   * Add a negative log-likelihood loss operation to the model.
   *
   * Calculates the nll loss given a probability tensor over classes, and
   * a target tensor containing class labels.
   *
   * \param args Vector of input tensor ids: probability and tensor.
   * \param reduction Type of reduction to perform on the individual losses.
   * \param ignoreIndex Optional class index to ignore in loss calculation.
   * \param inputIsLogProbability Specifies if the input tensor contains
   *                              log-probabilities or raw probabilities
   *                              (false, default).
   * \param debugContext Optional debug context.
   * \return The name of the result tensor.
   */

  TensorId nllloss(const std::vector<TensorId> &args,
                   const ReductionType reduction = ReductionType::Mean,
                   const nonstd::optional<int> ignoreIndex = nonstd::nullopt,
                   bool inputIsLogProbability              = false,
                   const DebugContext &debugContext        = {});

  /**
   * Add an identity loss operation to the model.
   *
   * Calculates the loss using the identity operator.
   *
   * \param args Vector of input tensor ids.
   * \param reduction Type of reduction to perform on the individual losses.
   * \param debugContext Optional debug context.
   * \return The name of the result tensor
   */
  TensorId identityloss(const std::vector<TensorId> &args,
                        const ReductionType reduction    = ReductionType::Mean,
                        const DebugContext &debugContext = {});

  /**
   * Add a connectionist temporal classification (CTC) loss operation to the
   * model.
   *
   * With T being maximum input length, N being batch size, C being number of
   * classes, S being a maximum target length, this op calculates the CTC loss
   * for a logarithmised probabilities tensor with shape [T, N, C], a class
   * target tensor with shape [N, S], an input lengths tensor [N] and a target
   * lengths tensor [N].
   *
   * Note that C includes a blank class (default=0). The probabilities tensor
   * is padded as required. Target sequences are also padded and are
   * populated with values less than equal to C, not including the blank class,
   * up to their respective target lengths. Note that target lengths cannot
   * exceed input lengths.
   *
   * \param args [log_probs,targets,input_lengths,target_lengths]
   * \param reduction Type of reduction to perform on the individual losses
   * \param blank The integer representing the blank class.
   * \param debugContext Optional debug context
   * \return The name of the result tensor
   */
  TensorId ctcloss(const std::vector<TensorId> &args,
                   const ReductionType reduction    = ReductionType::Mean,
                   const unsigned blank             = 0,
                   const std::string &outDataType   = "UNDEFINED",
                   const DebugContext &debugContext = {});

  // Additional version of ctcloss that returns both output tensors. The second
  // tensor is a tensor that is only expected to be used internally by CtcGradOp
  // to calculate gradients during the backwards pass. This prototype is useful
  // for poptorch but need not be included in doxygen documentation.
  std::vector<TensorId>
  _ctcloss(const std::vector<TensorId> &args,
           const ReductionType reduction    = ReductionType::Mean,
           const unsigned blank             = 0,
           const std::string &outDataType   = "UNDEFINED",
           const DebugContext &debugContext = {});

  /**
   * Add a shaped dropout operation to the model.
   *
   * Applies a shaped dropout to the input tensor. This operator requires a
   * shape parameter that is used to define the shape of the dropout mask so
   * that strongly correlated features in the input tensor can be preserved.
   * The provided shape must be broadcastable to the input tensor.  Note that
   * this operation targets the poprand library function of the same name.
   *
   * \param args Vector of input tensor ids.
   * \param shape Shape of dropout mask. Must be broadcastable to the input.
   * \param ratio Probability of dropping an input feature (default = 0.5).
   * \param name Optional identifier for operation.
   * \return The name of the result tensor.
   */
  TensorId shapeddropout(const std::vector<TensorId> &args,
                         const std::vector<int64_t> &shape,
                         float ratio                      = 0.5f,
                         const DebugContext &debugContext = {});

  /**
   * Add an \c atan2 operation to the model.
   *
   * Returns the element-wise angle theta as a tensor, -pi < theta <= pi, such
   * that for two input tensors x and y and given r != 0,
   * x = r cos theta, and
   * y = r sin theta, element-wise.
   *
   * In the case of x > 0, theta = arctan(y/x).
   *
   * \param args Vector of input tensor ids: [y, x].
   * \param name Optional identifier for operation.
   * \return The name of the result tensor containing element wise theta values.
   */
  TensorId atan2(const std::vector<TensorId> &args,
                 const DebugContext &debugContext = {});

  /**
   * Add \c expm1 operation to the model.
   * It computes exp(x) - 1.
   * Calculates the element-wise exponential of the input tensor and
   * subtracts one.
   *
   * \param args Vector of input tensor ids.
   * \param name Optional identifier for operation.
   * \return The name of the result tensor.
   */
  TensorId expm1(const std::vector<TensorId> &args,
                 const DebugContext &debugContext = {});

  /**
   * Add \c log1p operation to the model.
   * It computes log(x + 1).
   * This calculates the element-wise logarithm of the
   * input tensor plus one.
   *
   * \param args Vector of input tensor ids.
   * \param name Optional identifier for operation.
   * \return The name of the result tensor.
   */
  TensorId log1p(const std::vector<TensorId> &args,
                 const DebugContext &debugContext = {});

  /**
   * Add reshape operation to the model.
   * Reshape the input tensor.
   * This reshape takes the shape to reshape into as an attribute
   * instead of a tensor input as the ONNX reshape op.
   *
   * \param arg Vector with single input tensor id.
   * \param shape The shape of the output Tensor. The output Tensor
   * must contain the same number of elements as the input Tensor.
   * \param name Optional identifier for operation.
   * \return The name of the result tensor.
   */
  TensorId reshape(const TensorId &arg,
                   const Attributes::Ints &shape,
                   const DebugContext &debugContext = {});

  /**
   * Add fmod operation to the model.
   *
   * This is equivalent to C's fmod function. The result has the same sign as
   * the dividend.
   *
   * \param args Input tensors.
   * \return Computes the element-wise remainder of division. The remainder has
   * the same sign as the dividend.
   */
  TensorId fmod(const std::vector<TensorId> &args,
                const DebugContext &debugContext = {});

  /**
   * Add remainder operation to the model.
   *
   * This is equivalent to Python's modulo operator %. The result has the same
   * sign as the divisor.
   * \param args Input tensors.
   * \return Computes the element-wise remainder of division. The remainder has
   * the same sign as the divisor.
   */
  TensorId remainder(const std::vector<TensorId> &args,
                     const DebugContext &debugContext = {});

  /**
   * Add a reverse operator to the model.
   *
   * Reverse, or 'flip', the tensor along the specified dimensions
   *
   * \param args Input tensors.
   * \param dimensions Dimensions along which to reverse the tensor. If this is
   *        empty then this is equivalent to the identity operator
   * \return The name of the result tensor.
   */
  TensorId reverse(const std::vector<TensorId> &args,
                   const std::vector<int64_t> &dimensions,
                   const DebugContext &debugContext = {});

  /**
   * Add abort operation to the model.
   *
   * The operation can be conditional or unconditional.
   * \param args Optional input tensor to test condition
   */
  void abort(const std::vector<TensorId> &args,
             const DebugContext &debugContext = {});

  /**
   * Add a bitwise NOT operation to the model.
   *
   * The operation computes the bitwise NOT of a given integer tensor.
   * \param args Input tensor of type integer.
   * \return The name of the result tensor.
   */
  TensorId bitwisenot(const std::vector<TensorId> &args,
                      const DebugContext &debugContext = {});

  /**
   * Add a bitwise AND operation to the model.
   *
   * The operation computes the bitwise AND of given two integer tensors.
   * \param args Two broadcastable input tensors of type integer.
   * \return The name of the result tensor.
   */
  TensorId bitwiseand(const std::vector<TensorId> &args,
                      const DebugContext &debugContext = {});

  /**
   * Add a bitwise OR operation to the model.
   *
   * The operation computes the bitwise OR of given two integer tensors.
   * \param args Two broadcastable input tensors of type integer.
   * \return The name of the result tensor.
   */
  TensorId bitwiseor(const std::vector<TensorId> &args,
                     const DebugContext &debugContext = {});

  /**
   * Add a bitwise XOR operation to the model.
   *
   * The operation computes the bitwise XOR of given two integer tensors.
   * \param args Two broadcastable input tensors of type integer.
   * \return The name of the result tensor.
   */
  TensorId bitwisexor(const std::vector<TensorId> &args,
                      const DebugContext &debugContext = {});

  /**
   * Add a bitwise XNOR operation to the model.
   *
   * The operation computes the bitwise XNOR of given two integer tensors.
   * \param args Two broadcastable input tensors of type integer.
   * \return The name of the result tensor.
   */
  TensorId bitwisexnor(const std::vector<TensorId> &args,
                       const DebugContext &debugContext = {});

  /*
   * Add reducemedian operation to the model.
   *
   * It computes the median values along the specified axes. In the case of even
   * number of elements, the lower of the two medians is selected. By default,
   * the input tensor is reduced over all axes. Additionally, the operation also
   * returns the indices of found median values in the reduction axis. If
   * reduction is performed over multiple axes, the indices is a flattened index
   * over the reduced axes, similar to numpy.ndarray.flat. The index may not be
   * the first occurrence of the median value found in the input tensor.
   *
   * \param args Vector with single input tensor id.
   * \param axes Axes over which the reduction is performed.
   * \param keepdims If true, the result tensors are of equal size as the input,
   *        but with reduction axes of size 1. Otherwise, the reduction
   *        axes are squeezed and the result tensors have fewer dimensions
   *        compared to the input.
   * \param debugContext Optional debug information.
   * \return The names of the two result tensors, one for median values and one
   *         for indices.
   */
  std::vector<TensorId> reducemedian(
      const std::vector<TensorId> &args,
      const nonstd::optional<std::vector<int64_t>> &axes = nonstd::nullopt,
      int64_t keepdims                                   = 1,
      const DebugContext &debugContext                   = {});
};

/**
 * An interface for a Builder, used for creating ONNX graphs.
 */
class Builder {
  Builder();

public:
  /**
   * Return a Builder for a graph which is nested inside this Builder's graph.
   */

  Builder &createSubgraphBuilder();

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
   * Add a new input tensor to the model.
   *
   * \param tensorInfo The shape and type of the input tensor.
   * \param debugContext Optional debug information.
   * \return The unique name of the input tensor.
   */
  TensorId addInputTensor(const TensorInfo &tensorInfo,
                          const popart::DebugContext &debugContext = {});

  /**
   * Add a new input tensor to the model.
   *
   * \param dataType The type of the input tensor.
   * \param shape The shape of the input tensor.
   * \param debugContext Optional debug information.
   * \return The unique name of the input tensor.
   */
  TensorId addInputTensor(const std::string &dataType,
                          const Shape &shape,
                          const popart::DebugContext &debugContext = {});

  /**
   * Add a new input tensor without a type or shape to the model.
   *
   * \param debugContext Optional debug information.
   * \return The unique name of the input tensor.
   */
  TensorId addUntypedInputTensor(const popart::DebugContext &debugContext = {});

  /**
   * Add a new named input tensor to the model.
   *
   * \param tensorId The identifier string of the input tensor. This identifier
   * must already exist in the parent GraphProto's name scope and must appear
   * topologically before this sub-graph.
   */
  void addInputTensorFromParentGraph(const TensorId &tensorId);

  /**
   * Add a new pre-initialized input tensor to the model.
   *
   * \param initData The initial data of the input tensor.
   * \param debugContext Optional debug information.
   * \return The unique name of the input tensor.
   */
  TensorId
  addInitializedInputTensor(const ConstVoidData &initData,
                            const popart::DebugContext &debugContext = {});

  /**
   * Adds one of the outputs from a node in the graph into the list of output
   * tensors.
   */
  void addOutputTensor(const TensorId &arg0);

  /**
   * Return the builder interface for ai.onnx opset 6.
   */
  AiOnnxOpset6 aiOnnxOpset6() { return AiOnnxOpset6(this->impl_); }

  /**
   * Return the builder interface for ai.onnx opset 7.
   */
  AiOnnxOpset7 aiOnnxOpset7() { return AiOnnxOpset7(this->impl_); }

  /**
   * Return the builder interface for ai.onnx opset 7.
   */
  AiOnnxOpset8 aiOnnxOpset8() { return AiOnnxOpset8(this->impl_); }

  /**
   * Return the builder interface for ai.onnx opset 9.
   */
  AiOnnxOpset9 aiOnnxOpset9() { return AiOnnxOpset9(this->impl_); }

  /**
   * Return the builder interface for ai.onnx opset 10.
   */
  AiOnnxOpset10 aiOnnxOpset10() { return AiOnnxOpset10(this->impl_); }

  /**
   * Return the builder interface for ai.onnx opset 11.
   */
  AiOnnxOpset11 aiOnnxOpset11() { return AiOnnxOpset11(this->impl_); }

  /**
   * Return the builder interface for ai.onnx.ml opset 1.
   */
  AiOnnxMlOpset1 aiOnnxMlOpset1() { return AiOnnxMlOpset1(this->impl_); }

  /**
   * Return the builder interface for ai.graphcore opset 1.
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
           const std::map<std::string, popart::any> &attributes,
           const DebugContext &debugContext = {});

  // Add a custom op to the model
  // provide the name of the output tensors to use
  void customOp(const OperatorIdentifier &opid,
                int opsetVersion,
                const std::vector<TensorId> &inputs,
                const std::vector<TensorId> &outputs,
                const std::map<std::string, popart::any> &attributes,
                const DebugContext &debugContext = {});

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

  void outputTensorLocation(const TensorId &nodeOutputName,
                            TensorLocation value) {
    addNodeAttribute(
        sExecutionPhaseAttribute, value.serialize(), {nodeOutputName});
  }

  void recomputeOutput(const TensorId &nodeOutputName, RecomputeType value) {
    addNodeAttribute(sExecutionPhaseAttribute,
                     static_cast<int64_t>(value),
                     {nodeOutputName});
  }

  /**
   * Enable/disable recomputation of the output of the node in the backward
   * pass.
   *
   * \param nodeOutputName Name of the output tensor of the ONNX node.
   * \param value If the recompute is enabled/disabled.
   */
  void recomputeOutputInBackwardPass(
      const TensorId &nodeOutputName,
      RecomputeType value = RecomputeType::Recompute) {
    addNodeAttribute(sRecomputeOutputAttribute,
                     static_cast<int64_t>(value),
                     {nodeOutputName});
  }

  /**
   * Enable/disable recomputation of the output of the node in the backward
   * pass.
   *
   * \param nodeOutputNames Names of the output tensors of the ONNX node.
   * \param value If the recompute is enabled/disabled.
   */
  void recomputeOutputInBackwardPass(
      const std::set<TensorId> &nodeOutputNames,
      RecomputeType value = RecomputeType::Recompute) {
    addNodeAttribute(sRecomputeOutputAttribute,
                     static_cast<int64_t>(value),
                     nodeOutputNames);
  }

  /**
   * Return whether the given node will have its output recomputed in the
   * backward pass.
   *
   * \param nodeOutputName Name of the output tensor of the ONNX node used to
   *                        find the node in the ONNX model.
   */
  bool getRecomputeOutputInBackwardPass(const TensorId &nodeOutputName) {
    return getBoolNodeAttribute(sRecomputeOutputAttribute, {nodeOutputName});
  }

  /**
   * Return whether the given node will have its output recomputed in the
   * backward pass.
   *
   * \param nodeOutputNames Names of the output tensors of the ONNX node used to
   *                        find the node in the ONNX model.
   */
  bool
  getRecomputeOutputInBackwardPass(const std::set<TensorId> &nodeOutputNames) {
    return getBoolNodeAttribute(sRecomputeOutputAttribute, nodeOutputNames);
  }

  /**
   * Add checkpoint operations to the model.
   *
   * This is the same as an identity but is recomputeType Checkpoint by default.
   *  Use this to checkpoint a subset of an operation's output tensors.
   *
   * \param nodeOutputNames Tensors to checkpoint.
   * \return The checkpointed tensors.
   */
  std::vector<TensorId>
  checkpointOutput(const std::vector<TensorId> &nodeOutputNames);

  /**
   * Set the virtual graph that computes the given node.  Applies when creating
   * a graph for a multi-IPU configuration.
   *
   * \param nodeOutputName Name of the output tensor of the ONNX node.
   * \param value The index of the virtual graph that computes this node.
   */
  void virtualGraph(const TensorId &nodeOutputName, int64_t value = 0) {
    addNodeAttribute(sVirtualGraphAttribute, value, {nodeOutputName});
  }

  /**
   * Set the execution phase that computes the given node.
   * \param nodeOutputName Name of the output tensor of the ONNX node.
   * \param value The index of the virtual graph that computes this node.
   */
  void executionPhase(const TensorId &nodeOutputName, int64_t value = 0) {
    addNodeAttribute(sExecutionPhaseAttribute, value, {nodeOutputName});
  }

  void pipelineStage(const TensorId &nodeOutputName, int64_t value) {
    addNodeAttribute(sPipelineStageAttribute, value, {nodeOutputName});
  }

  void pipelineStage(const std::set<TensorId> &nodeOutputNames, int64_t value) {
    addNodeAttribute(sPipelineStageAttribute, value, nodeOutputNames);
  }

  void excludePatterns(const TensorId &nodeOutputName,
                       const std::vector<std::string> &patternNames) {
    addNodeAttribute(sExcludePatternsAttribute, patternNames, {nodeOutputName});
  }

  void excludePatterns(const std::set<TensorId> &nodeOutputNames,
                       const std::vector<std::string> &patternNames) {
    addNodeAttribute(sExcludePatternsAttribute, patternNames, nodeOutputNames);
  }

  /**
   * Set the settings for matmuls that should be serialized. This option
   * will split a matmul into separate smaller matmuls that will be executed in
   * series. This will also serialize the grad operations if training.
   *
   *
   * \param nodeOutputNames Name of the output matmul tensors of the ONNX node.
   * \param mode Which dimension of the mat mul to serialize on.
   * \param factor The number of serialised matmuls, must be a factor of the
   * dimensions to serialise on.
   *
   */
  void setSerializeMatMul(const std::set<TensorId> &nodeOutputNames,
                          std::string mode,
                          int64_t factor,
                          bool keep_precision) {
    if (mode == sSerializeMatMulMode_InputChannels ||
        mode == sSerializeMatMulMode_OutputChannels ||
        mode == sSerializeMatMulMode_ReducingDim) {
      addNodeAttribute(sSerializeMatMulModeAttribute, mode, nodeOutputNames);
      addNodeAttribute(
          sSerializeMatMulFactorAttribute, factor, nodeOutputNames);
      addNodeAttribute(sSerializeMatMulPrecisionAttribute,
                       static_cast<int64_t>(keep_precision),
                       nodeOutputNames);
    } else if (mode != sSerializeMatMulMode_None) {
      throw error("Unsupported mat mul serialization mode '{}'. Supported "
                  "modes are '{}', '{}', '{}' or '{}'",
                  mode,
                  sSerializeMatMulMode_InputChannels,
                  sSerializeMatMulMode_ReducingDim,
                  sSerializeMatMulMode_OutputChannels,
                  sSerializeMatMulMode_None);
    }
  }

  /**
   * Set the partials type for the given node. Used on the convolution op.
   *
   * \param nodeOutputName Name of the output tensor of the ONNX node.
   * \param partialsType The type for the partials. Can be either FLOAT or HALF.
   */
  void setPartialsType(const TensorId &nodeOutputName,
                       const std::string partialsType);

  /**
   * Get the partials type for the given node.
   *
   * \param nodeOutputName Name of the output tensor of the ONNX node.
   */
  std::string getPartialsType(const TensorId &nodeOutputName);
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
   * Set the available memory for the given node. Used on the convolution op.
   *
   * \param nodeOutputName Name of the output tensor of the ONNX node.
   * \param availableMemoryProportion The available memory proportion 0 < x
   * <= 1.
   */
  void setAvailableMemoryProportion(const TensorId &nodeOutputName,
                                    const float availableMemoryProportion);
  /**
   * Set an attribute that will be set on all subsequent operations.
   */
  void setAttribute(const std::string &attribute, popart::any value);

  /**
   * Get an attribute that has been set for all subsequent operations.
   */
  popart::any getAttribute(const std::string attribute) const;

  bool hasAttribute(const std::string &attribute) const;

  /**
   * Unset an attribute that will be set on all subsequent operations.
   */
  void clearAttribute(const std::string &attribute);

  /**
   * Check if an attribute is set.
   */
  bool hasAttribute(const std::string &attribute);

  /**
   * Get the current attribute value.
   */
  popart::any getAttribute(const std::string &attribute);

  /**
   * A convenience function for getting the pipeline stage attribute.
   */
  int64_t getPipelineStage() const;

  /**
   * A convenience function for getting the execution phase attribute.
   */
  int64_t getExecutionPhase() const;

  /**
   * A convenience function for getting the virtual graph attribute.
   */
  int64_t getVirtualGraph() const;

  /**
   * Set the virtual graph that computes the given node.  Applies when creating
   * a graph for a multi-IPU configuration.
   *
   * \param nodeOutputNames Names of the output tensors of the ONNX node.
   * \param value The index of the virtual graph that computes this node.
   */
  void virtualGraph(const std::set<TensorId> &nodeOutputNames,
                    int64_t value = 0) {
    addNodeAttribute(sVirtualGraphAttribute, value, nodeOutputNames);
  }

  void executionPhase(const std::set<TensorId> &nodeOutputNames,
                      int64_t value = 0) {
    addNodeAttribute(sExecutionPhaseAttribute, value, nodeOutputNames);
  }

  /**
   * Add an attribute to the ONNX node which is uniquely identified by the
   * outputs.
   * This functions will throw an exception if it can't find the unique
   * node or the attribute already exists.
   *
   * \param attributeName The name of the attribute to add.
   * \param attributeValue An \c int64_t value of the attribute to add.
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
   * \param attributeValue A \c std::vector<int64_t> value of the attribute to
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
   * \param attributeValue A \c float value of the attribute to add.
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
   * \param attributeValue The \c std::vector<float> value of the attribute to
   *                       add.
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
   * \param attributeValue A \c std::string value of the attribute to add.
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
   * \param attributeValue A \c std::vector<std::string> value of the attribute
   *                       to add.
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
   * \param attributeValue A bool value of the attribute to add.
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
   * \param attributeValue A constant tensor initializer.
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
   * Get the \c int64_t value of the attribute for the ONNX node.
   * This functions will throw an exception if it can't find the unique
   * node or the attribute does not exist or it has not been set to the
   * \c int64_t type.
   *
   * \param attributeName The name of the attribute to find.
   * \param nodeOutputNames Names of the output tensors of the ONNX node used to
   *                        find the node in the ONNX model.
   * \return Value of the attribute.
   */
  int64_t getInt64NodeAttribute(const std::string &attributeName,
                                const std::set<TensorId> &nodeOutputNames);

  /**
   * Get the \c std::vector<int64_t> value of the attribute for the ONNX node.
   * This functions will throw an exception if it can't find the unique
   * node or the attribute does not exist or it has not been set to the
   * \c std::vector<int64_t> type.
   *
   * \param attributeName The name of the attribute to find.
   * \param nodeOutputNames Names of the output tensors of the ONNX node used to
   *                        find the node in the ONNX model.
   * \return Value of the attribute.
   */
  std::vector<int64_t>
  getInt64VectorNodeAttribute(const std::string &attributeName,
                              const std::set<TensorId> &nodeOutputNames);

  /**
   * Get the \c float value of the attribute for the ONNX node.
   * This functions will throw an exception if it can't find the unique
   * node or the attribute does not exist or it has not been set to the
   * \c float type.
   *
   * \param attributeName The name of the attribute to find.
   * \param nodeOutputNames Names of the output tensors of the ONNX node used to
   *                        find the node in the ONNX model.
   * \return Value of the attribute.
   */
  float getFloatNodeAttribute(const std::string &attributeName,
                              const std::set<TensorId> &nodeOutputNames);

  /**
   * Get the \c std::vector<float> value of the attribute for the ONNX node.
   * This functions will throw an exception if it can't find the unique
   * node or the attribute does not exist.
   *
   * \param attributeName The name of the attribute to find.
   * \param nodeOutputNames Names of the output tensors of the ONNX node used to
   *                        find the node in the ONNX model.
   * \return Value of the attribute.
   */
  std::vector<float>
  getFloatVectorNodeAttribute(const std::string &attributeName,
                              const std::set<TensorId> &nodeOutputNames);

  /**
   * Get the \c std::string value of the attribute for the ONNX node.
   * This functions will throw an exception if it can't find the unique
   * node or the attribute does not exist or it has not been set to the
   * \c std::string type.
   *
   * \param attributeName The name of the attribute to find.
   * \param nodeOutputNames Names of the output tensors of the ONNX node used to
   *                        find the node in the ONNX model.
   * \return Value of the attribute.
   */
  std::string getStringNodeAttribute(const std::string &attributeName,
                                     const std::set<TensorId> &nodeOutputNames);

  /**
   * Get the \c std::vector<std::string> value of the attribute for the ONNX
   * node.
   * This functions will throw an exception if it can't find the unique
   * node or the attribute does not exist.
   *
   * \param attributeName The name of the attribute to find.
   * \param nodeOutputNames Names of the output tensors of the ONNX node used to
   *                        find the node in the ONNX model.
   * \return Value of the attribute.
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

  int64_t getExecutionPhase(const TensorId &nodeOutputName) {
    return getInt64NodeAttribute(sExecutionPhaseAttribute, {nodeOutputName});
  }

  int64_t getExecutionPhase(const std::set<TensorId> &nodeOutputNames) {
    return getInt64NodeAttribute(sExecutionPhaseAttribute, nodeOutputNames);
  }

  /**
   * Retrieve the ONNX serialized ModelProto.
   *
   * \return A serialized ONNX ModelProto.
   */
  std::string getModelProto() const;

  /**
   * Save the builder's ONNX ModelProto into the builder and validate it.
   *
   * \param fn The name of a file containing an ONNX model protobuf.
   */
  void saveModelProto(const std::string &fn);

  /**
   * Save tensor data externally.
   *
   * The model data cannot exceed 2GB - the maximum size of a Protobuf
   * message. To avoid this, for large models ONNX tensor data can be
   * saved separately.
   *
   * \param ids The names of tensors whose data is to be saved externally.
   * \param fn The name of a file containing the binary tensor data. This
   *        can be an absolute or relative path. If a relative path, when
   *        the ONNX model is saved, external tensor data will be written
   *        to a path relative to your current working directory.
   */
  void saveInitializersExternally(const std::vector<TensorId> &ids,
                                  const std::string &fn);

  /**
   * Return a list of ONNX graph input tensor ids.
   *
   * \return A vector of input tensor names.
   */
  std::vector<TensorId> getInputTensorIds() const;

  /**
   * Return a list of ONNX graph output tensor ids.
   *
   * \return A vector of output tensor names.
   */
  std::vector<TensorId> getOutputTensorIds() const;

  /**
   * Return a list of ONNX graph value tensor ids.
   *
   * These tensors are stored in the `value_info` section
   * of the ONNX GraphProto structure.
   *
   * \return A vector of output tensor names.
   */
  std::vector<TensorId> getValueTensorIds() const;

  /**
   * Return a list of ONNX graph initialized tensor ids.
   *
   * These tensors are stored in the `initialized` section of the ONNX
   * GraphProto structure..
   *
   * \return A vector of tensor names.
   */
  std::vector<TensorId> getTrainableTensorIds() const;

  /**
   * Return an ONNX graph tensor shape, from either the input,
   * output, or value_info lists in the GraphProto.
   *
   * \param id Tensor id.
   * \return A vector of tensor dimensions.
   */
  std::vector<int64_t> getTensorShape(const TensorId id);

  /**
   * Returns true if the ONNX tensor is in the initializer list
   * of the GraphProto.
   *
   * \param id Tensor id.
   * \return A boolean.
   */
  bool isInitializer(const TensorId id) const;

  /**
   * Return an ONNX graph tensor type as a lower case string, from either
   * the input, output, or value_info lists in the GraphProto.
   *
   * \param id Tensor id.
   * \return A lower case string of tensor type.
   */
  std::string getTensorDtypeString(const TensorId id);

  /**
   * Return a tensor type from either
   * the input, output, or value_info lists in the GraphProto.
   *
   * \param id Tensor id.
   * \return A tensor type.
   */
  DataType getTensorDataType(const TensorId id);

  /**
   * Push a name onto the name scope stack.
   *
   * The names of tensors and nodes added to the ONNX graph will be prefixed
   * with a concatenation of the names in the name stack.
   */
  void pushNameScope(const std::string &name);

  /**
   * Remove the last entry in the name scope stack.
   */
  void popNameScope();

  /**
   * Get the current namescope stack using the default delimiter.
   *
   * \param name Optional string to concatenate to the end of the stack
   * \return A string of the concatenated namescope stack.
   */
  std::string getNameScope(const std::string &name = "") const;

  /**
   * Specifies a graph name.
   *
   * \param name String to name the graph.
   */
  void setGraphName(const std::string &name);

  /**
   * Sets the parent graph of this builder.
   *
   * \param parent the builder to become a parent.
   */
  void setParent(Builder *parent);

  /**
   * Returns the parent graph of this graph or null if there is no parent.
   */
  Builder *getParent() const;

  /**
   * Returns true if this builder represents a subgraph.
   */
  bool hasParent() const { return parent == nullptr; }

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
  std::map<int, std::unique_ptr<Builder>> children;
  int nChildren{0};

  Builder *parent;
};

} // namespace popart

#endif // GUARD_BUILDER_HPP
