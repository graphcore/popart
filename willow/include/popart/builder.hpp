// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_BUILDER_HPP_
#define POPART_WILLOW_INCLUDE_POPART_BUILDER_HPP_

#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include <popart/dataflow.hpp>
#include <popart/debugcontext.hpp>
#include <popart/domainopset.hpp>
#include <popart/names.hpp>
#include <popart/op.hpp>
#include <popart/op/collectives/collectives.hpp>
#include <popart/op/scatterreduce.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/tensorlocation.hpp>
#include <popart/variablesettings.hpp>
#include <popart/vendored/any.hpp>
#include <popart/vendored/optional.hpp>

// Include the generated builder.gen.hpp code
#include "popart/attributes.hpp"
#include "popart/builder.gen.hpp"
#include "popart/error.hpp"
#include "popart/logging.hpp"

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
class ConstVoidData;
struct OperatorIdentifier;
class CommGroup;

enum class DataType;
enum class RecomputeType;

/// Class that represents the AI ONNX ML opset.
class AiOnnxMlOpset1 : public DomainOpSet {

protected:
  using DomainOpSet::impl;

  int getOpsetVersion() const override { return 1; }

public:
  /**
   * Constructor for the AiOnnxMlOpset1 class.
   *
   * \param impl_ A pointer to an implementation of the Builder class.
   */
  AiOnnxMlOpset1(std::unique_ptr<BuilderImpl> &impl_) : DomainOpSet(impl_) {}
};

/// Class that represents the AI Graphcore opset.
class AiGraphcoreOpset1 : public DomainOpSet {
  // Builds an op for specified bitwise operator id.
  TensorId bitwiseGenericOp(const OperatorIdentifier &opid,
                            const std::vector<TensorId> &args,
                            const DebugContext &debugContext = {});

protected:
  using DomainOpSet::impl;

  int getOpsetVersion() const override { return 1; }

public:
  /**
   * Constructor for the AiGraphcoreOpset1 class.
   *
   * \param impl_ A pointer to an implementation of the Builder class.
   */
  AiGraphcoreOpset1(std::unique_ptr<BuilderImpl> &impl_) : DomainOpSet(impl_) {}

  /**
   * Copies a tensor to an initalised tensor (variable).
   *
   * This is used to update an initalised tensor (a variable created using
   * addInitializedInputTensor()) which retains its value between iterations, by
   * setting the value to the value of another tensor (the updater). The purpose
   * is to manually update the tensor in use cases for variables other than
   * trained parameters (weights) or tensors used by other ops.
   *
   * \param args A vector of the input tensor ids containing the tensor to be
   *      updated, `tensor` and the tensor containing the values for the update,
   *      `updater` as [`tensor`, `updater`].
   * \param debugContext Optional debug information.
   * \return An alias to the updated variable: to ensure correct ordering of
   *   the updated variable, you should use this variable for any op which
   *   should operate on the updated variable.
   */
  TensorId copyvarupdate(const std::vector<TensorId> &args,
                         const DebugContext &debugContext = {});

  /**
   * Add a batch normalization operation to the model. This version uses N-1
   * as the population size for calculating running variance (like PyTorch).
   * https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html
   *
   * Whereas, the Onnx version uses N.
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#BatchNormalization
   *
   * \param args List of input tensor ids
   * \param num_outputs The number of output tensor ids
   * \param epsilon The 'epsilon' attribute
   * \param momentum The 'momentum' attribute
   * \param name Optional identifier for the operation
   * \return A list of normalized output tensors
   */
  std::vector<TensorId>
  batchnormalization(const std::vector<TensorId> &args,
                     unsigned num_outputs,
                     float epsilon                            = 1e-05f,
                     float momentum                           = 0.9f,
                     const popart::DebugContext &debugContext = {});

  /**
   * Add a group normalization operation to the model.
   *
   * This is a Poplar extension.
   *
   * The group will be created from a strided input.
   *
   * \param args A vector of input tensor ids for input data `x`, scale `scale`,
   *      and bias `bias` as [`x`, `scale`, `bias`].
   * \param num_groups The number of groups to separate the channels into.
   * \param epsilon The epsilon value to use to avoid division by zero.
   * \param debugContext Optional debug information.
   * \return A vector of output tensor ids for output data `y`, the mean `mean`
   * and the variance `var` as [`y`, `mean`, `var`].
   */
  std::vector<TensorId>
  groupnormalization(const std::vector<TensorId> &args,
                     int64_t num_groups,
                     float epsilon                    = 1e-05f,
                     const DebugContext &debugContext = {});

  // clang-format off
// Need long lines for URLs
  /**
   * Add a multi-convolution operation to the model.
   *
   * Using this multi-convolution API ensures that the convolutions are
   * executed in parallel on the device.
   *
   * Functionally, a multi-convolution is equivalent to a series of single
   * convolutions. Using this multi-convolution API is always equivalent to
   * calling the single-convolution API (conv) once for each argument.
   *
   * For example, calling:
   * ```
   *     A0 = conv({X0, W0, B0})
   *     A1 = conv({X1, W1})
   *```
   * is functionally equivalent to calling:
   *```
   *     {A0, A1} = multiconv({{X0, W0, B0}, {X1, Q1}).
   *```
   * It is possible that any two convolutions cannot be executed in parallel
   * due to topological constraints. For example, the following:
   *```
   *     B = conv({A, W0});
   *     C = B + A
   *     D = conv({C, W1});
   *```
   * cannot be converted to:
   *```
   *     {B, D} = multiconv({{A, W0}, {C, W1}}).
   *```
   * Note that it is not possible to create such a cycle by adding a
   * multi-convolution with this API.
   *
   * Calls to multiconv() are mapped to
   * poplar::poplin::multiconv::convolution().
   *
   * All input vectors must be either empty, or equal in length to
   * the number of convolutions. Note that groups for each convolution are
   * automatically inferred from the shapes of the data and weight inputs.
   *
   * \param tensors List of tensor ids for input tensors for data, weights and
   *      biases as [`data`, `weight`,`bias`] for each convolution. `bias` is
   *      optional.
   * \param dilations The dilations attributes for each convolution.
   * \param inDilations The input dilations attributes for each convolution.
   * \param pads The pads for each convolution.
   * \param outPads The output padding for each convolution.
   * \param strides The strides for each convolution.
   * \param availableMemoryProportions The available memory proportions per
   *     convolution, each [0, 1).
   * \param partialsTypes The partials type per convolution.
   * \param planType Run convolutions in parallel or series.
   * \param perConvReservedTiles The number of tiles to reserve per convolution
   *     when planning.
   * \param cycleBackOff Cycle back-off proportion, [0, 1).
   * \param enableConvDithering Enable convolution dithering per convolution. If
   *     `true`, then convolutions with different parameters will be laid out
   *     from different tiles in an effort to improve tile balance in models.
   * \param debugContext Optional debug information.
   *
   * \return A vector of tensor ids of the output tensor from each convolution.
   *
   * \sa <a href="https://docs.graphcore.ai/projects/available-memory/">Optimising Temporary Memory Usage for Convolutions and Matmuls on the IPU</a> for some practical examples of using `availableMemoryProportion`.
   */
  // clang-format on
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
            const std::vector<int64_t> enableConvDithering   = {},
            const DebugContext &debugContext                 = {});

  /**
   * Add a sub-sample operation to the model.
   *
   * This is a Poplar extension.
   *
   * If multiple tensors are provided, the strides will be applied to them all.
   *
   * \param args A vector of tensor ids to sub-sample.
   * \param strides The strides to use.
   * \param debugContext Optional debug information.
   * \return The tensor id of the result tensor.
   */
  TensorId subsample(const std::vector<TensorId> &args,
                     const std::vector<int64_t> &strides,
                     const DebugContext &debugContext = {});

  /**
   * Add a print tensor operation to the model.
   *
   * This is a Poplar extension.
   *
   * \param args A vector of tensor ids to print.
   * \param print_gradient Indicates whether the gradient tensor(s) associated
   *      with the input tensor(s) are also printed. If 1, the gradient
   *      tensor(s) are also printed, otherwise the gradient tensor(s) are not
   *      printed.
   * \param debugContext Optional debug information.
   * \param title An optional title to print.
   * \param summariseThreshold (default 1000) If the number of elements of the
   * tensor exceeds this threshold the output will be summarised. Only the edge
   * elements will be displayed with an ellipsis indicating skipped elements.
   * A value of 0 will disable summarisation.
   * \param edgeItems (default 3) number of edge elements to include at the
   * beginning and end when summarisation is enabled
   * \param maxLineWidth (default 75) lines longer than this limit will be split
   * across multiple lines. A value of 0 will disable line splitting.
   * \param digits (default 8) number of digits to display. For integers this
   * limit can be exceeded if any number is large enough. For floating points
   * this does not include the exponent. The number of digits is used in
   * conjunction analysis of the tensor to determine the width of each element
   * to align all elements when printed. A value of 0 disables this analysis
   * and each elements will be printed in an unaligned format.
   * \param floatFormat (default 0=Auto) determines the floating point format to
   * use. 0=auto, 1=fixed, 2=scientific 3=none. Automatic mode determines the
   * appropriate format based on the data. If `digits==0` this option is
   * disregarded and the floatFormat is set to `none`.
   * \param separator (default space) character used to delininate values.
   * \param openBracket (default square bracket) character used to open a
   * tensor.
   * \param closeBracket (default square bracket) character used to close a
   * tensor.
   * \return The tensor id of the result tensor.
   */
  TensorId printtensor(const std::vector<TensorId> &args,
                       int64_t print_gradient           = 1,
                       const DebugContext &debugContext = {},
                       const std::string &title         = {},
                       const int summariseThreshold     = 1000,
                       const int edgeItems              = 3,
                       const int maxLineWidth           = 75,
                       const int digits                 = 8,
                       const int floatFormat            = 0,
                       const char separator             = ' ',
                       const char openBracket           = '[',
                       const char closeBracket          = ']');

  /**
   * Add a no-op operation to the model.
   *
   * \param args A vector of input tensor ids.
   * \param debugContext Optional debug information.
   * \return The tensor id of the result tensor.
   */
  TensorId nop(const std::vector<TensorId> &args,
               const DebugContext &debugContext = {});

  /**
   * Add a scale operation to the model.
   *
   * This is a Poplar extension.
   *
   * \param args A vector of input tensor ids.
   * \param scale The scale to apply.
   * \param debugContext Optional debug information.
   * \return The tensor id of the result tensor.
   */
  TensorId scale(const std::vector<TensorId> &args,
                 float scale,
                 const DebugContext &debugContext = {});
  /**
   * Add a scaled add operation to the model.
   *
   * The scaled add operation takes the form:
   * ```
   *  X = scale0 * T0 + scale1 * T1
   * ```
   * where \c scale0 is the scale factor to be applied to tensor \T0 and
   * \c scale1 is the scale factor to be applied to tensor \T1.
   *
   * \param args A vector of input tensor ids: [T0, T1, scale0, scale1].
   * \param scale0 The scale to apply (if no \c scale0 tensor is supplied).
   * \param scale1 The scale to apply (if no \c scale1 tensor is supplied).
   * \param debugContext Optional debug information.
   * \return The tensor id of the result tensor.
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
   * \param args A vector of input tensor ids.
   * \param debugContext Optional debug information.
   * \return The tensor id of the result tensor.
   */
  TensorId gelu(const std::vector<TensorId> &args,
                const DebugContext &debugContext = {});

  /**
   * Add a detach operation to the model.
   *
   * \param args A vector of input tensor ids.
   * \param debugContext Optional debug information.
   * \return The tensor id of the result tensor.
   */
  TensorId detach(const std::vector<TensorId> &args,
                  const DebugContext &debugContext = {});

  // clang-format off
  /**
   * Add a depth-to-space operation to the model.
   *
   * This allows DepthToSpace_11 to be targeted from earlier opsets.
   *
   * The purpose of a depth-to-space operation, also known as pixel shuffling,
   * is to rearrange data from the depth (channels) dimension into the spatial
   * (width and height) dimensions. It is an efficient means of learning
   * upsampling alongside mixing convolution with bilinear interpolation and
   * using transpose convolution.
   *
   * \sa <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#DepthToSpace">ONNX DepthToSpace operator</a>.
   *
   * \param args A vector containing a single tensor id of the input tensor
   *      of shape [`N`,`C`,`H`,`W`], where `N` is the batch axis, `C` is the
   *      channel or depth, `H` is the height and `W` is the width.
   * \param blocksize The size of the blocks to be moved. If the input is
   *      [`N`, `C`, `H`, `W`] and the blocksize is `B`, the output will be
   *      [`N`, `C/(B*B)`, `H*B`, `W*B`].
   * \param mode Specifies how the data is rearranged:
   *    * "DCR" (Default): depth-column-row order
   *    * "CRD": column-row-depth order
   * \param debugContext Optional debug information.
   * \return A tensor which is a rearrangement of the input tensor.
   */
  // clang-format on
  TensorId depthtospace(const std::vector<TensorId> &args,
                        int64_t blocksize,
                        const std::string &mode          = "DCR",
                        const DebugContext &debugContext = {});

  // clang-format off
  // Need long lines for URL
  /**
   * Add a rounding operation to the model.
   *
   * This allows \c Round_11 to be targeted from earlier opsets.
   *
   * \sa <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#Round">ONNX Round operator</a>.
   *
   * \param args A vector of input tensor ids.
   * \param debugContext Optional debug information.
   * \return The normalized output tensor ids.
   */
  // clang-format on
  TensorId round(const std::vector<TensorId> &args,
                 const DebugContext &debugContext = {});

  /**
   * Add an init operation to the model.
   *
   * \param shape The shape of the tensor to initialise.
   * \param data_type The data type to initialise tensor with. The value is the
   *      integer attribute taken from the DataType enum.
   * \param init_type The mode of the tensor initialisation. The value is the
   *      integer attribute taken from the InitType enum.
   * \param batch_axis Batch axis specifies the axis that the batches are split
   *      along and is a literal integer.
   * \param debugContext Optional debug information.
   * \return The tensor id of the result tensor.
   */
  TensorId init(Attributes::Ints shape,
                Attributes::Int data_type,
                Attributes::Int init_type,
                Attributes::Int batch_axis,
                const DebugContext &debugContext = {});

  /**
   * Add an init operation to the model.
   *
   * \param shape The shape of the tensor to initialise.
   * \param data_type The data type to initialise tensor with. The value is the
   *      integer attribute taken from the DataType enum.
   * \param init_type The mode of the tensor initialisation. The value is the
   *      integer attribute taken from the InitType enum.
   * \param debugContext Optional debug information.
   * \return The tensor id of the result tensor.
   */
  TensorId init(Attributes::Ints shape,
                Attributes::Int data_type,
                Attributes::Int init_type,
                const DebugContext &debugContext = {});

  /**
   * Add a dynamic slice operation to the model.
   *
   * Creates a new slice tensor, \c slice, at offset position, \c offset, in a
   * tensor, \c tensor.
   * For example:
   * ```
   *  slice = tensor[offset]
   *```
   * \param args A vector of input tensor ids: [tensor, offset].
   * \param axes The axes along which to slice.
   * \param sizes The size of the slice along each axis.
   * \param noOverlap Indicates whether the slice regions overlap or not. If 1,
   *      slice regions do not overlap, otherwise they do overlap.
   * \param debugContext Optional debug information.
   * \return The tensor id of the result tensor.
   */
  TensorId dynamicslice(const std::vector<TensorId> &args,
                        Attributes::Ints axes,
                        Attributes::Ints sizes,
                        Attributes::Int noOverlap,
                        const DebugContext &debugContext = {});
  /**
   * Add a dynamic update operation to the model.
   *
   * Creates a copy of a tensor, \c tensor, and updates the elements of the
   * copied tensor at offset position, \c offset, with the elements contained
   * in the slice tensor, \c slice,
   * For example:
   * ```
   *  out = tensor
   *  out[offset] = slice
   *```
   * \param args A vector of input tensor ids: [tensor, offset, slice].
   * \param axes The axes along which to update.
   * \param sizes The size of the slice along each axis.
   * \param noOverlap Indicates whether the updates overlap or not. If 1,
   *      the updates do not overlap, otherwise they do overlap.
   * \param debugContext Optional debug information.
   * \return The tensor id of the result tensor.
   */
  TensorId dynamicupdate(const std::vector<TensorId> &args,
                         Attributes::Ints axes,
                         Attributes::Ints sizes,
                         Attributes::Int noOverlap,
                         const DebugContext &debugContext = {});
  /**
   * Add a dynamic zero operation to the model.
   *
   * Creates a copy of a tensor, \c tensor, with a slice tensor at offset
   * position, \c offset set to zero.
   * For example:
   * ```
   *  out = tensor
   *  out[offset] = 0.0
   * ```
   * \param args A vector of input tensor ids: [tensor, offset].
   * \param axes The axes along which to zero elements.
   * \param sizes The size of the slice along each axis.
   * \param debugContext Optional debug information.
   * \return The tensor id of the result tensor.
   */
  TensorId dynamiczero(const std::vector<TensorId> &args,
                       Attributes::Ints axes,
                       Attributes::Ints sizes,
                       const DebugContext &debugContext = {});
  /**
   * Add a dynamic add operation to the model.
   *
   * Creates a copy of a tensor, \c tensor, with a slice tensor, \c slice,
   * added at an offset position, \c offset.
   * For example:
   *```
   *  out = tensor
   *  out[offset] += slice
   *```
   * \param args A vector of input tensor ids: [`tensor`, `offset`, `slice`].
   * \param axes The axes along which to add the slice.
   * \param sizes The size of the slice along each axis.
   * \param debugContext Optional debug information.
   * \return The tensor id of the result tensor.
   */
  TensorId dynamicadd(const std::vector<TensorId> &args,
                      Attributes::Ints axes,
                      Attributes::Ints sizes,
                      const DebugContext &debugContext = {});

  /**
   * Slice a 2D tensor based on offsets.
   *
   * The outermost dimension is sliced. For the following:
   *  * `source` is the source tensor.
   *  * `destination` is the destination tensor.
   *  * `N` is the number of elements to copy.
   *  * `sourceOffset` is the first element read from the source tensor.
   *  * `destinationOffset` is the first element written to in the destination
   *     tensor.
   * Then, for each entry in `N`, `sourceOffset` and `destinationOffset`:
   *```
   * destination[destinationOffset:destinationOffset+N][...] =
   * source[sourceOffset:sourceOffset+N][...]
   * ```
   * Entries after the first `N==0` may be ignored.
   * Unreferenced elements of `destination` are zeroed if `zeroUnused` is
   * set. The same output element should not be written by multiple inputs.
   *
   * `source` and `destination` must have rank greater than or equal to 2. The
   * outer dimension
   * is sliced; the product of the inner dimensions must match. `sourceOffset`,
   * `destinationOffset` and `N` must be 1-dimensional and of the same size.
   * For example:
   *
   *```
   * N = [1, 1, 1]
   * sourceOffset = [0, 2, 4]
   * destinationOffset = [0, 1, 2]
   *```
   * \param args A vector of input tensor ids for the following tensors
   *      [`source`, `destination`, `N`, `sourceOffset`, `destinationOffset`].
   * \param zeroUnused Determines whether to zero unreferenced `destination`
   *      elements. If 1, the unreferenced elements are zeroed, otherwise they
   *      are not zeroed.
   * \param debugContext Optional debug information.
   */
  TensorId sequenceslice(const std::vector<TensorId> &args,
                         Attributes::Int zeroUnused,
                         const DebugContext &debugContext = {});

  /**
   * Add a call operation to the model.
   *
   * This is a Poplar extension, to expose manual code re-use to
   * the builder.
   *
   * \param args A vector of input tensor ids.
   * \param callee The subgraph to call into.
   * \param debugContext Optional debug information.
   * \return A vector of tensors; the subgraph outputs.
   */
  std::vector<TensorId> call(const std::vector<TensorId> &args,
                             unsigned num_outputs,
                             const Builder &callee,
                             const DebugContext &debugContext = {});

  /**
   * DEPRECATED: Add a replicated allreduce operation to the model.
   *
   * This is a Poplar extension, to expose manual code re-use to
   * the builder.
   *
   * \param args A vector of input tensor ids to reduce across.
   * \param commGroup GCL CommGroup parameter.
   * \param debugContext Optional debug information.
   * \return The tensor id of the result tensor.
   */
  TensorId replicatedallreduce(
      const std::vector<TensorId> &args,
      const nonstd::optional<std::vector<int64_t>> &commGroup = nonstd::nullopt,
      const DebugContext &debugContext                        = {});

  /**
   * Add a replicated allreduce operation to the model.
   *
   * This is a Poplar extension, to expose manual code re-use to
   * the builder.
   *
   * \param args A vector of input tensor ids to reduce across
   * \param collectiveOperator A Graphcore Communication Library (GCL)
   *      collective operator.
   * \param commGroup A GCL CommGroup parameter.
   * \param debugContext Optional debug information
   * \return The tensor id of the result tensor.
   */
  TensorId replicatedallreduce(
      const std::vector<TensorId> &args,
      const nonstd::optional<CollectiveOperator> &collectiveOperator =
          nonstd::nullopt,
      const nonstd::optional<CommGroup> &commGroup = nonstd::nullopt,
      const DebugContext &debugContext             = {});

  /**
   * Add a replicated reduce-scatter operation to the model.
   *
   * This is a Poplar extension, to expose manual code re-use to
   * the builder.
   *
   * \param args A vector of input tensor ids to reduce across.
   * \param collectiveOperator A Graphcore Communication Library (GCL)
   *      collective operator.
   * \param commGroup A GCL CommGroup parameter.
   * \param debugContext Optional debug information.
   * \return The tensor id of the result tensor.
   */
  TensorId replicatedreducescatter(
      const std::vector<TensorId> &args,
      const nonstd::optional<CollectiveOperator> &collectiveOperator =
          nonstd::nullopt,
      const nonstd::optional<CommGroup> &commGroup = nonstd::nullopt,
      const DebugContext &debugContext             = {});

  /**
   * Add an \c l1 loss operation to the model.
   *
   * Calculates the mean absolute error between each element in the input with
   * a zero target.
   *
   * \param args A vector of input tensor ids.
   * \param lambda The scale factor of the L1 loss.
   * \param reduction The type of reduction to perform on the individual losses.
   * \param debugContext Optional debug information.
   * \return The tensor id of the result tensor.
   */
  TensorId l1loss(const std::vector<TensorId> &args,
                  const float lambda,
                  const ReductionType reduction    = ReductionType::Mean,
                  const DebugContext &debugContext = {});

  /**
   * Add a negative log-likelihood loss operation to the model.
   *
   * Calculates the negative log likelihood (NLL) loss given a probability
   * tensor over classes, and a target tensor containing class labels.
   *
   * \param args A vector of input tensor ids: probability and tensor.
   * \param reduction The type of reduction to perform on the individual losses.
   * \param ignoreIndex Optional class index to ignore in loss calculation.
   * \param inputIsLogProbability If `true` the input tensor contains
   *     log-probabilities, otherwise raw probabilities. Default = `false`.
   * \param debugContext Optional debug information.
   * \return The tensor id of the result tensor.
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
   * \param args A vector of input tensor ids.
   * \param reduction The type of reduction to perform on the individual losses.
   * \param debugContext Optional debug information.
   * \return The tensor id of the result tensor.
   */
  TensorId identityloss(const std::vector<TensorId> &args,
                        const ReductionType reduction    = ReductionType::Mean,
                        const DebugContext &debugContext = {});

  /**
   * Add a tensor remap operation to the model.
   *
   * Changes the tensor layout to conform to the downstream consumers, which
   * means the consumers can read the tensor without having to rearrange it.
   *
   * \param args The tensor id of the tensor to remap. This is a single tensor
   *      that should be copied to a new tensor with a tensor layout conforming
   *      to the downstream consumer.
   * \param remap_type The type of remap to perform on the forward/backward
   *      pass. Backward pass remapping requires the op to exist in the
   *      IR before autodiff. The value is the integer attribute value of the
   *      enum TensorRemapType.
   * \param debugContext Optional debug information.
   * \return The tensor id of the result tensor.
   */
  TensorId tensorremap(const std::vector<TensorId> &args,
                       Attributes::Int remap_type,
                       const DebugContext &debugContext = {});

  /**
   * Add a connectionist temporal classification (CTC) loss operation to the
   * model.
   *
   * With maximum input length `T`, batch size `N`, number of classes `C` and
   * maximum target length `S`, this op calculates the CTC loss for a
   * logarithmised probabilities tensor with shape [`T`, `N`, `C`], a class
   * target tensor with shape [`N`, `S`], an input lengths tensor [`N`] and a
   * target lengths tensor [`N`].
   *
   * Note that `C` includes a blank class (default=0). The probabilities tensor
   * is padded as required. Target sequences are also padded and are populated
   * with values less than or equal to `C`, not including the blank class, up to
   * their respective target lengths. Note that target lengths cannot exceed
   * input lengths.
   *
   * \param args A vector of input tensor ids [`log_probs`,`targets`,
   *      `input_lengths`, `target_lengths`].
   * \param reduction The type of reduction to perform on the individual losses.
   * \param blank The integer representing the blank class.
   * \param outDataType The data type of the output tensors. Default =
   *      `UNDEFINED`.
   * \param zeroInfinity If `true` infinite losses and the associated
   *      gradients are zeroed-out. Default = `false`.
   * \param debugContext Optional debug information
   * \return The tensor id of the result tensor.
   */
  TensorId ctcloss(const std::vector<TensorId> &args,
                   const ReductionType reduction    = ReductionType::Mean,
                   const unsigned blank             = 0,
                   const std::string &outDataType   = "UNDEFINED",
                   const bool zeroInfinity          = false,
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
           const bool zeroInfinity          = false,
           const DebugContext &debugContext = {});

  /**
   * Add a connectionist temporal classification (CTC) beam search decoder
   * operation to the model.
   *
   * Calculate the most likely \p topPaths labels and their probabilities given
   * the input \p logProbs with lengths \p dataLengths.
   *
   * \param args A vector of input tensor ids. These are [`logProbs`,
   *     `dataLengths`], where `logProbs` is of shape [`maxTime`, `batchSize`, *
   *     `numClasses`], and `dataLengths` is of shape [`batchSize`].
   * \param blank The integer representing the blank class.
   * \param beamWidth The number of beams to use when decoding.
   * \param topPaths The number of most likely decoded paths to return, must be
   *      less than or equal to \p beamWidth.
   * \param  debugContext Optional debug information.
   *
   * \return The names of the result tensors. These are [`labelProbs,
   *      `labelLengths`, `decodedLabels`], where
   *      `labelProbs` is of shape
   *      [`batchSize`, `topPaths`], `labelLengths` is of shape [`batchSize`,
   *      `topPaths`], and `decodedLabels` is of shape [`batchSize`,
   *      `topPaths`, `maxTime`].
   */
  std::vector<TensorId>
  ctcbeamsearchdecoder(const std::vector<TensorId> &args,
                       unsigned blank                   = 0,
                       unsigned beamWidth               = 100,
                       unsigned topPaths                = 1,
                       const DebugContext &debugContext = {});

  /**
   * Add a shaped dropout operation to the model.
   *
   * Applies a shaped dropout to the input tensor. This operator requires a
   * shape parameter that is used to define the shape of the dropout mask so
   * that strongly correlated features in the input tensor can be preserved.
   * The provided shape must be broadcastable to the input tensor.  Note that
   * this operation targets the `poprand` library function of the same name.
   *
   * \param args A vector of input tensor ids.
   * \param shape The shape of dropout mask. This must be broadcastable to the
   *      input.
   * \param ratio The probability of dropping an input feature. Default = 0.5.
   * \param debugContext Optional debug information.
   * \return The tensor id of the result tensor.
   */
  TensorId shapeddropout(const std::vector<TensorId> &args,
                         const std::vector<int64_t> &shape,
                         float ratio                      = 0.5f,
                         const DebugContext &debugContext = {});

  /**
   * Add an \c atan2 operation to the model.
   *
   * Returns the element-wise angle theta as a tensor.
   * For \f$ -\pi < \theta \le \pi \f$, such
   * that for two input tensors \f$x\f$ and \f$y\f$ and given \f$ r \ne 0 \f$,
   * then \f$ x = r \cos\theta \f$, and \f$ y = r \sin\theta \f$, element-wise.
   *
   * In the case of \f$ x > 0 \f$ , \f$ \theta = arctan(y/x)\f$ .
   *
   * \param args A vector of input tensor ids: [`y`, `x`].
   * \param debugContext Optional debug information.
   * \return The tensor id of the result tensor.
   */
  TensorId atan2(const std::vector<TensorId> &args,
                 const DebugContext &debugContext = {});

  /**
   * Add a \c expm1 operation to the model.
   *
   * This calculates the element-wise exponential of the input tensor and
   * subtracts one: \f$ exp(x) - 1 \f$.
   *
   * \param args A vector of input tensor ids.
   * \param debugContext Optional debug information.
   * \return The tensor id of the result tensor.
   */
  TensorId expm1(const std::vector<TensorId> &args,
                 const DebugContext &debugContext = {});

  /**
   * Add a \c log1p operation to the model.
   *
   * This calculates the element-wise logarithm of the input tensor plus one:
   * \f$ log(x + 1) \f$.
   *
   * \param args A vector of input tensor ids.
   * \param name Optional identifier for operation.
   * \param debugContext Optional debug information.
   * \return The tensor id of the result tensor.
   */
  TensorId log1p(const std::vector<TensorId> &args,
                 const DebugContext &debugContext = {});

  /**
   * Add a reshape operation to the model.
   *
   * This reshapes an input tensor.
   * This reshape takes the target shape as an attribute
   * instead of a tensor input as for the ONNX reshape op.
   *
   * \param arg The tensor id of the input tensor.
   * \param shape The shape of the output tensor. The output tensor
   *     must contain the same number of elements as the input tensor.
   * \param debugContext Optional debug information.
   * \return The tensor id of the result tensor.
   */
  TensorId reshape(const TensorId &arg,
                   const Attributes::Ints &shape,
                   const DebugContext &debugContext = {});

  /**
   * Add an `fmod` operation to the model.
   *
   * This is equivalent to the C `fmod` function. The result has the same sign
   * as the dividend.
   *
   * \param args A vector of input tensor ids.
   * \param debugContext Optional debug information.
   * \return Computes the element-wise remainder of division. The remainder has
   *     the same sign as the dividend.
   */
  TensorId fmod(const std::vector<TensorId> &args,
                const DebugContext &debugContext = {});

  /**
   * Add a remainder operation to the model.
   *
   * This is equivalent to Python's modulo operator `%`. The result has the same
   * sign as the divisor.
   * \param args A vector of input tensor ids.
   * \param debugContext Optional debug information.
   * \return Computes the element-wise remainder of division. The remainder has
   *     the same sign as the divisor.
   */
  TensorId remainder(const std::vector<TensorId> &args,
                     const DebugContext &debugContext = {});

  /**
   * Add a reverse operator to the model.
   *
   * This reverses or flips the tensor along the specified dimensions.
   *
   * \param args A vector of input tensor ids.
   * \param dimensions The dimensions along which to reverse the tensor. If
   *      this is empty then this is equivalent to the identity operator.
   * \param debugContext Optional debug information.
   * \return The tensor id of the reversed tensor.
   */
  TensorId reverse(const std::vector<TensorId> &args,
                   const std::vector<int64_t> &dimensions,
                   const DebugContext &debugContext = {});

  /**
   * Add a slice to the model.
   *
   * This version of slice uses the `starts`, `ends` and `axes` attributes
   * rather than tensor inputs. This reduces the number of ops as constant
   * tensors are treated as ops while attributes are not.
   *
   * \param args A vector of input tensor ids.
   * \param ends The `ends` attribute.
   * \param starts The `starts` attribute.
   * \param axes The `axes` attribute.
   * \param debugContext Optional debug information.
   * \return The normalized output tensor id.
   */
  TensorId slice(const std::vector<TensorId> &args,
                 const std::vector<int64_t> &ends,
                 const std::vector<int64_t> &starts,
                 const std::vector<int64_t> &axes = std::vector<int64_t>(),
                 const popart::DebugContext &debugContext = {});

  /**
   * Add a packedDataBlock operator to the model.
   *
   * Unpack packed sequences of data and call the callback function on the
   * unpacked sequences.
   *
   * \param args A vector of input tensor ids.
   * \param maxSequenceLengths The maximum length of a sequence in each of the
   *     data inputs.
   * \param resultSize The size of the first dimension of the
   *     result tensor.
   * \param callbackBatchSize The number of batches to pass
   *     to the callback.
   * \param callback The callback function.
   * \param debugContext Optional debug information.
   * \return The tensor id of the result tensor.
   */
  TensorId packedDataBlock(const std::vector<TensorId> &args,
                           const std::vector<int64_t> &maxSequenceLengths,
                           int64_t resultSize,
                           int64_t callbackBatchSize,
                           const Builder &callback,
                           const DebugContext &debugContext = {});

  /**
   * Add an abort operation to the model.
   *
   * The operation can be conditional or unconditional.
   * \param args A vector of input tensor ids.
   * \param debugContext Optional debug information.
   */
  void abort(const std::vector<TensorId> &args,
             const DebugContext &debugContext = {});

  /**
   * Add a bitwise NOT operation to the model.
   *
   * The operation computes the bitwise NOT of an integer tensor.
   * \param args An input tensor of type integer.
   * \param debugContext Optional debug information.
   * \return The tensor id of the result tensor.
   */
  TensorId bitwisenot(const std::vector<TensorId> &args,
                      const DebugContext &debugContext = {});

  /**
   * Add a bitwise AND operation to the model.
   *
   * The operation computes the bitwise AND of two integer tensors.
   * \param args Two broadcastable input tensors of type integer.
   * \param debugContext Optional debug information.
   * \return The tensor id of the result tensor.
   */
  TensorId bitwiseand(const std::vector<TensorId> &args,
                      const DebugContext &debugContext = {});

  /**
   * Add a bitwise OR operation to the model.
   *
   * The operation computes the bitwise OR of two integer tensors.
   * \param args Two broadcastable input tensors of type integer.
   * \param debugContext Optional debug information.
   * \return The tensor id of the result tensor.
   */
  TensorId bitwiseor(const std::vector<TensorId> &args,
                     const DebugContext &debugContext = {});

  /**
   * Add a bitwise XOR operation to the model.
   *
   * The operation computes the bitwise XOR of two integer tensors.
   * \param args Two broadcastable input tensors of type integer.
   * \param debugContext Optional debug information.
   * \return The tensor id of the result tensor.
   */
  TensorId bitwisexor(const std::vector<TensorId> &args,
                      const DebugContext &debugContext = {});

  /**
   * Add a bitwise XNOR operation to the model.
   *
   * The operation computes the bitwise XNOR of two integer tensors.
   * \param args Two broadcastable input tensors of type integer.
   * \param debugContext Optional debug information.
   * \return The tensor id of the result tensor.
   */
  TensorId bitwisexnor(const std::vector<TensorId> &args,
                       const DebugContext &debugContext = {});

  /**
   * Add reducemedian operation to the model.
   *
   * This method computes the median values along the specified axes. In the
   * case of an even number of elements, the lower of the two medians is
   * selected. By default, the input tensor is reduced over all axes.
   * Additionally, the operation also returns the indices of found median values
   * in the reduction axis. If reduction is performed over multiple axes, the
   * indices are "flattened" over the reduced axes, similar to
   * `numpy.ndarray.flat`. The index may not be the first occurrence of the
   * median value found in the input tensor.
   *
   * \param args A vector with a single input tensor id.
   * \param axes The axes over which the reduction is performed.
   * \param keepdims If 1, the result tensors are of equal size as the
   *      input, but with reduction axes of size 1. Otherwise, the reduction
   *      axes are squeezed and the result tensors have fewer dimensions
   *      compared to the input. Default = 1.
   * \param debugContext Optional debug information.
   * \return The names of the two result tensors, one for median values and one
   *     for indices.
   */
  std::vector<TensorId> reducemedian(
      const std::vector<TensorId> &args,
      const nonstd::optional<std::vector<int64_t>> &axes = nonstd::nullopt,
      int64_t keepdims                                   = 1,
      const DebugContext &debugContext                   = {});

  /**
   * Add a scatterreduce operation to the model.
   *
   * Reduces all the values from the source tensor `src` at the indices
   * specified along the given axis by `index`. In some frameworks this is also
   * known as a split-apply-combine operation as well as a reduce or aggregate
   * by key.  In this analogy the `src` input is the data we are splitting and
   * the `indices` define the groups for the reduction operation.
   *
   * In pseudocode the operator can be expressed as:
   * ```
   *  for i in range(axis_size):
   *      output[i] = reduce(src[index == i])
   * ```
   * where the looping over output indices is implicitly handled by poplar.
   *
   * \param args A vector of tensor ids as [`src`, `index`].
   * \param axis_size The size of the reduced axis.
   * \param axis The axis to reduce along. Default = -1.
   * \param reduction The type of reduction to apply. Default =
   *      `ScatterReduction::Sum`.
   * \param debugContext Optional debug information.
   * \return The tensor id of the result tensor.
   */
  TensorId scatterreduce(const std::vector<TensorId> &args,
                         Attributes::Int axis_size,
                         Attributes::Int axis       = -1,
                         ScatterReduction reduction = ScatterReduction::Sum,
                         const DebugContext &debugContext = {});

  /**
   * Add a swish operation to the model.
   *
   * The operation computes the swish activation function, also known
   * as the SiLU activation.
   *
   * \param args A vector with a single input tensor id.
   * \param debugContext Optional debug information.
   * \return The tensor id of the result tensor.
   */
  TensorId swish(const std::vector<TensorId> &args,
                 const DebugContext &debugContext = {});

  /**
   * Add an incrementmod operation to the model.
   *
   * The operation is of the form `y = (x + increment) % modulus`.
   *
   * \param args A vector with a single input tensor id.
   * \param increment A scalar increment
   * \param modulus A scalar modulus
   * \param debugContext Optional debug information.
   * \return The tensor id of the result tensor.
   */
  TensorId incrementmod(const std::vector<TensorId> &args,
                        Attributes::Float increment,
                        Attributes::Float modulus,
                        const DebugContext &debugContext = {});
};

/**
 * An interface for a Builder, used for creating ONNX graphs.
 */
class Builder {
  /// Constructor for the Builder class.
  Builder();

public:
  /**
   * Create a builder for a graph which is nested inside this builder's graph.
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
   *      file containing an ONNX model protobuf.
   */
  static std::unique_ptr<Builder>
  createFromOnnxModel(const std::string &modelProtoOrFilename);

  /// Destructor for the Builder class.
  ~Builder();

  /**
   * Add a new input tensor to the model.
   *
   * \param tensorInfo The shape and data type of the input tensor.
   * \param debugContext Optional debug information.
   * \return The tensor id of the input tensor.
   */
  TensorId addInputTensor(const TensorInfo &tensorInfo,
                          const popart::DebugContext &debugContext = {});

  /**
   * Add a new input tensor to the model.
   *
   * \param dataType The data type of the input tensor.
   * \param shape The shape of the input tensor.
   * \param debugContext Optional debug information.
   * \return The tensor id of the input tensor.
   */
  TensorId addInputTensor(const std::string &dataType,
                          const Shape &shape,
                          const popart::DebugContext &debugContext = {});

  /**
   * Add a new input tensor to the model.
   *
   * \param tensorInfo The shape and data type of the input tensor.
   * \param InputSettings Settings for \p TileSet and \p ExchangeStrategy.
   * \param debugContext Optional debug information.
   * \return The tensor id of the input tensor.
   */
  TensorId addInputTensor(const TensorInfo &tensorInfo,
                          const InputSettings &settings,
                          const popart::DebugContext &debugContext = {});

  /**
   * Add a new input tensor to the model.
   *
   * \param dataType The data type of the input tensor.
   * \param shape The shape of the input tensor.
   * \param InputSettings Settings for \p TileSet and \p ExchangeStrategy.
   * \param debugContext Optional debug information.
   * \return The tensor id of the input tensor.
   */
  TensorId addInputTensor(const std::string &dataType,
                          const Shape &shape,
                          const InputSettings &settings,
                          const popart::DebugContext &debugContext = {});

  /**
   * Add a new input tensor without a type or shape to the model.
   *
   * \param debugContext Optional debug information.
   * \return The tensor id of the input tensor.
   */
  TensorId addUntypedInputTensor(const popart::DebugContext &debugContext = {});

  /**
   * Add a new named input tensor (from the parent graph) to the model.
   *
   * \param tensorId The identifier string of the input tensor. This identifier
   *      must already exist in the name scope of the parent `GraphProto` and
   *      must appear topologically before this sub-graph.
   */
  void addInputTensorFromParentGraph(const TensorId &tensorId);

  /**
   * Add a new pre-initialized input tensor to the model.
   *
   * \param initData The initial data of the input tensor.
   * \param debugContext Optional debug information.
   * \return The tensor id of the input tensor.
   */
  TensorId
  addInitializedInputTensor(const ConstVoidData &initData,
                            const popart::DebugContext &debugContext = {});

  /**
   * Add a new pre-initialized input tensor to the model.
   *
   * \param initData The initial data of the input tensor.
   * \param variableSettings The settings that determine how variables are
   *      retrieved from replicas.
   * \param debugContext Optional debug information.
   * \return The tensor id of the input tensor.
   */
  TensorId
  addInitializedInputTensor(const ConstVoidData &initData,
                            const VariableSettings &variableSettings,
                            const popart::DebugContext &debugContext = {});

  /**
   * Add an output tensor from a node in the graph into the list of output
   * tensors.
   *
   * \param arg0 The tensor id of the output tensor to be added.
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
   * Return the builder interface for ai.onnx opset 8.
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

  /**
   * Return the output tensors from a custom op added to the model.
   *
   *\param opid The id of the operator.
   *\param opsetVersion The version of the opset.
   *\param inputs The tensor ids of the A vector of input tensor ids.
   *\param numOutputs The number of output tensors.
   *\param attributes The map of attributes and their values to be added.
   *\param debugContext Optional debug information.
   * \returns The output tensors.
   */
  std::vector<TensorId>
  customOp(const OperatorIdentifier &opid,
           int opsetVersion,
           const std::vector<TensorId> &inputs,
           const unsigned numOutputs,
           const std::map<std::string, popart::any> &attributes,
           const DebugContext &debugContext = {});

  /**
   * Add a custom op to the model.
   *
   *\param opid The id of the operator.
   *\param opsetVersion The version of the opset.
   *\param inputs The tensor ids of the A vector of input tensor ids.
   *\param outputs The tensor ids of the output tensors.
   *\param attributes The map of attributes and their values to be added.
   *\param debugContext Optional debug information.
   */
  void customOp(const OperatorIdentifier &opid,
                int opsetVersion,
                const std::vector<TensorId> &inputs,
                const std::vector<TensorId> &outputs,
                const std::map<std::string, popart::any> &attributes,
                const DebugContext &debugContext = {});

  /**
   * Add a constant and a reshape a tensor using the provided domain.
   *
   *\param t The builder interface.
   *\param args The tensor ids of the tensors to be updated.
   *\param shape The shape information to be used.
   *\param name (Optional) The name of the updated tensor. Default: None.
   * \return The tensor id of the updated tensor.
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
   * Set a value for the output tensor location attribute.
   *
   * \param nodeOutputName The tensor id of the output tensor of the ONNX node.
   * \param value The location of the tensor.
   */
  void outputTensorLocation(const TensorId &nodeOutputName,
                            TensorLocation value) {
    addNodeAttribute(
        sOutputTensorLocationAttribute, value.serialize(), {nodeOutputName});
  }

  /**
   * Enable recomputation of the output of the node in the backward pass.
   *
   * \param nodeOutputName The tensor id of the output tensor of the ONNX node.
   * \param value (Optional) The type of the recompute.
   */
  void recomputeOutput(const TensorId &nodeOutputName, RecomputeType value) {
    addNodeAttribute(sRecomputeOutputAttribute,
                     static_cast<int64_t>(value),
                     {nodeOutputName});
  }

  /**
   * Enable or disable recomputation of the output of the node in the backward
   * pass.
   *
   * \param nodeOutputName The tensor id of the output tensor of the ONNX node.
   * \param value (Optional) The type of the recompute. Default:
   *      `RecomputeType::Recompute`.
   */
  void recomputeOutputInBackwardPass(
      const TensorId &nodeOutputName,
      RecomputeType value = RecomputeType::Recompute) {
    addNodeAttribute(sRecomputeOutputAttribute,
                     static_cast<int64_t>(value),
                     {nodeOutputName});
  }

  /**
   * Enable or disable recomputation of the output of the node in the backward
   * pass.
   *
   * \param nodeOutputNames The tensor ids of the output tensors of the ONNX
   *      node.
   * \param value (Optional) The type of the recompute. Default:
   *      `RecomputeType::Recompute`.
   */
  void recomputeOutputInBackwardPass(
      const std::set<TensorId> &nodeOutputNames,
      RecomputeType value = RecomputeType::Recompute) {
    addNodeAttribute(sRecomputeOutputAttribute,
                     static_cast<int64_t>(value),
                     nodeOutputNames);
  }

  /**
   * Check if a node will have its output recomputed in the backward pass.
   *
   * \param nodeOutputNames The tensor ids of the output tensors of the
   *     ONNX node used to find the node in the ONNX model.
   * \returns `true` if the output will be recomputed; `false` otherwise.
   */
  bool getRecomputeOutputInBackwardPass(const TensorId &nodeOutputName) {
    return getBoolNodeAttribute(sRecomputeOutputAttribute, {nodeOutputName});
  }

  /**
   * Check if a node will have its output recomputed in the backward pass.
   *
   * \param nodeOutputNames The tensor ids of the output tensors of the
   *     ONNX node used to find the node in the ONNX model.
   * \returns `true` if the output will be recomputed; `false` otherwise.
   */
  bool
  getRecomputeOutputInBackwardPass(const std::set<TensorId> &nodeOutputNames) {
    return getBoolNodeAttribute(sRecomputeOutputAttribute, nodeOutputNames);
  }

  /**
   * Add checkpoint operations to the model.
   *
   * This is the same as an identity op but RecomputeType is `Checkpoint`
   * by default.
   * Use this to checkpoint a subset of an operation's output tensors.
   *
   * \param nodeOutputNames The tensors to checkpoint.
   * \return The checkpointed tensors.
   */
  std::vector<TensorId>
  checkpointOutput(const std::vector<TensorId> &nodeOutputNames);

  /**
   * Set the virtual graph that computes the given node.
   *
   * Applies when creating a graph for a multi-IPU configuration.
   *
   * \param nodeOutputName Name of the output tensor of the ONNX node.
   * \param value The index of the virtual graph that computes this node.
   *      Default=0.
   */
  void virtualGraph(const TensorId &nodeOutputName, int64_t value = 0) {
    addNodeAttribute(sVirtualGraphAttribute, value, {nodeOutputName});
  }

  /**
   * Set the execution phase that computes the given node.
   *
   * Applies when creating a graph for a multi-IPU configuration.
   *
   * \param nodeOutputName The tensor id of the output tensor of the ONNX node.
   * \param value The index of the virtual graph that computes this node.
   *      Default=0.
   */
  void executionPhase(const TensorId &nodeOutputName, int64_t value = 0) {
    addNodeAttribute(sExecutionPhaseAttribute, value, {nodeOutputName});
  }

  /**
   * Set the value on the pipeline stage attribute.
   *
   * \param nodeOutputName The tensor id of the output tensor of the ONNX node.
   * \param value The value to be set.
   */
  void pipelineStage(const TensorId &nodeOutputName, int64_t value) {
    addNodeAttribute(sPipelineStageAttribute, value, {nodeOutputName});
  }

  /**
   * Set the value on the pipeline stage attribute.
   *
   * \param nodeOutputNames The tensor ids of the output tensors of the ONNX
   *      node.
   * \param value The value to be set.
   */
  void pipelineStage(const std::set<TensorId> &nodeOutputNames, int64_t value) {
    addNodeAttribute(sPipelineStageAttribute, value, nodeOutputNames);
  }

  /**
   * Set the patterns to be excluded.
   *
   * \param nodeOutputName The tensor id of the output tensor of the ONNX node.
   * \param patternNames The vector of pattern names to be excluded.
   */
  void excludePatterns(const TensorId &nodeOutputName,
                       const std::vector<std::string> &patternNames) {
    addNodeAttribute(sExcludePatternsAttribute, patternNames, {nodeOutputName});
  }

  /**
   * Set the patterns to be excluded.
   *
   * \param nodeOutputNames The tensor ids of the output tensors of the ONNX
   *      node.
   * \param patternNames The vector of pattern names to be excluded.
   */
  void excludePatterns(const std::set<TensorId> &nodeOutputNames,
                       const std::vector<std::string> &patternNames) {
    addNodeAttribute(sExcludePatternsAttribute, patternNames, nodeOutputNames);
  }

  /**
   * Set the settings for matmuls that should be serialized.
   *
   * This option will split a matmul into separate smaller matmuls that will be
   * executed in series. This will also serialize the grad operations during
   * training.
   *
   * \param nodeOutputNames The tensor ids of the output matmul tensors of the
   *      ONNX node.
   * \param mode The dimension of the matmul to serialize on. Options
   *      are: 'input_channels', 'output_channels', 'reducing_dim', 'none'.
   * \param factor The number of serialised matmuls. This must be a factor of
   *      the dimensions to serialise on.
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
   * Set the partials type for the given node.
   *
   * This is used in the convolution op.
   *
   * \param nodeOutputName Name of the output tensor of the ONNX node.
   * \param partialsType The type for the partials. Options are: `FLOAT` or
   * `HALF`.
   */
  void setPartialsType(const TensorId &nodeOutputName,
                       const std::string partialsType);

  /**
   * Enable convolution dithering.
   *
   * \param nodeOutputName The tensor id of the output tensor of the ONNX node.
   * \param value The value to enable convolution. This should be 1 to enable
   *     convolution dithering and 0 otherwise.
   */
  void setEnableConvDithering(const TensorId &nodeOutputName, int64_t value);

  /**
   * Get the partials type for the given node.
   *
   * \param nodeOutputName The tensor id of the output tensor of the ONNX node.
   * \returns The partials type.
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

  // clang-format off
  // Need long lines for URL
  /**
   * Set the available memory proportion for the given node.
   *
   * This is used in the convolution op.
   *
   * \sa <a href="https://docs.graphcore.ai/projects/available-memory/">Optimising Temporary Memory Usage for Convolutions and Matmuls on the IPU</a> for some practical examples of using `availableMemoryProportion`.
   *
   * \param nodeOutputName Name of the output tensor of the ONNX node.
   * \param availableMemoryProportion The available memory proportion [0, 1).
   */
  // clang-format on
  void setAvailableMemoryProportion(const TensorId &nodeOutputName,
                                    const float availableMemoryProportion);

  // clang-format off
  // Need long lines for URL
  /**
   * Set the available memory proportion for the given node.
   *
   * This is used in the convolution op.
   *
   * \sa <a href="https://docs.graphcore.ai/projects/available-memory/">Optimising Temporary Memory Usage for Convolutions and Matmuls on the IPU</a> for some practical examples of using `availableMemoryProportion`

   *
   * \param nodeOutputNames The tensor ids of the output tensors of the
   *     ONNX node used to find the node in the ONNX model.
   * \param availableMemoryProportion The available memory proportion [0, 1).
   */
  // clang-format on
  void setAvailableMemoryProportion(const std::set<TensorId> &nodeOutputNames,
                                    const float availableMemoryProportion);

  /**
   * Set the value of an attribute that will be set on all subsequent
   * operations.
   *
   * \param attribute The name of the attribute to set.
   * \param value The value to set on the attribute.
   */
  void setAttribute(const std::string &attribute, popart::any value);

  /**
   * Get an attribute that has been set for all subsequent operations.
   *
   * \param attribute The name of the attribute to get.
   * \returns The attribute.
   */
  popart::any getAttribute(const std::string attribute) const;

  /**
   * Check if an attribute exists.
   *
   * \param attribute The name of the attribute to check.
   * \returns `true` if the attribute exists; `false` otherwise.
   */
  bool hasAttribute(const std::string &attribute) const;

  /**
   * Unset an attribute that will be set on all subsequent operations.
   *
   * \param attribute The name of the attribute to unset.
   */
  void clearAttribute(const std::string &attribute);

  /**
   * Check if an attribute is set.
   *
   * \param attribute The name of the attribute to check.
   * \returns `true` if the attribute is set; `false` otherwise.
   */
  bool hasAttribute(const std::string &attribute);

  /**
   * Get the attribute value.
   *
   * \param attribute The name of the attribute.
   * \returns The value of the attribute.
   */
  popart::any getAttribute(const std::string &attribute);

  /**
   * Get the pipeline stage attribute.
   *
   * \returns The pipeline stage.
   */
  int64_t getPipelineStage() const;

  /**
   * Get the execution phase attribute.
   *
   * \returns The execution phase.
   */
  int64_t getExecutionPhase() const;

  /**
   * Get the virtual graph attribute.
   *
   * \returns The virtual graph.
   */
  int64_t getVirtualGraph() const;

  /**
   * Set the virtual graph that computes the given node.
   *
   * Applies when creating a graph for a multi-IPU configuration.
   *
   * \param nodeOutputNames The tensor ids of the output tensors of the
   *     ONNX node used to find the node in the ONNX model.
   * \param value The index of the virtual graph that computes this node.
   */
  void virtualGraph(const std::set<TensorId> &nodeOutputNames,
                    int64_t value = 0) {
    addNodeAttribute(sVirtualGraphAttribute, value, nodeOutputNames);
  }

  /**
   * Set the execution phase.
   *
   * Applies when creating a graph for a multi-IPU configuration.
   *
   * \param nodeOutputNames The tensor ids of the output tensors of the
   *     ONNX node used to find the node in the ONNX model.
   * \param value The index of the virtual graph that computes this node.
   */
  void executionPhase(const std::set<TensorId> &nodeOutputNames,
                      int64_t value = 0) {
    addNodeAttribute(sExecutionPhaseAttribute, value, nodeOutputNames);
  }

  /**
   * Add an attribute to the ONNX node which is uniquely identified by the
   * output tensors.
   *
   * This function will throw an exception if it cannot find the unique
   * node or if the attribute already exists.
   *
   * \param attributeName The name of the attribute to add.
   * \param attributeValue An \c int64_t value of the attribute to add.
   * \param nodeOutputNames The tensor ids of the output tensors of the
   *     ONNX node used to find the node in the ONNX model.
   */
  void addNodeAttribute(const std::string &attributeName,
                        const int64_t &attributeValue,
                        const std::set<TensorId> &nodeOutputNames);

  /**
   * Add an attribute to the ONNX node which is uniquely identified by the
   * output tensors.
   *
   * This function will throw an exception if it cannot find the unique
   * node or if the attribute already exists.
   *
   * \param attributeName The name of the attribute to add.
   * \param attributeValue A \c std::vector<int64_t> value of the attribute to
   *      add.
   * \param nodeOutputNames The tensor ids of the output tensors of the
   *     ONNX node used to find the node in the ONNX model.
   */
  void addNodeAttribute(const std::string &attributeName,
                        const std::vector<int64_t> &attributeValue,
                        const std::set<TensorId> &nodeOutputNames);

  /**
   * Add an attribute to the ONNX node which is uniquely identified by the
   * output tensors.
   *
   * This function will throw an exception if it cannot find the unique
   * node or if the attribute already exists.
   *
   * \param attributeName The name of the attribute to add.
   * \param attributeValue A \c float value of the attribute to add.
   * \param nodeOutputNames The tensor ids of the output tensors of the
   *     ONNX node used to find the node in the ONNX model.
   */
  void addNodeAttribute(const std::string &attributeName,
                        const float &attributeValue,
                        const std::set<TensorId> &nodeOutputNames);

  /**
   * Add an attribute to the ONNX node which is uniquely identified by the
   * output tensors.
   *
   * This function will throw an exception if it cannot find the unique
   * node or if the attribute already exists.
   *
   * \param attributeName The name of the attribute to add.
   * \param attributeValue The \c std::vector<float> value of the attribute to
   *      add.
   * \param nodeOutputNames The tensor ids of the output tensors of the
   *     ONNX node used to find the node in the ONNX model.
   */
  void addNodeAttribute(const std::string &attributeName,
                        const std::vector<float> &attributeValue,
                        const std::set<TensorId> &nodeOutputNames);

  /**
   * Add an attribute to the ONNX node which is uniquely identified by the
   * output tensors.
   *
   * This function will throw an exception if it cannot find the unique
   * node or if the attribute already exists.
   *
   * \param attributeName The name of the attribute to add.
   * \param attributeValue A \c std::string value of the attribute to add.
   * \param nodeOutputNames The tensor ids of the output tensors of the
   *     ONNX node used to find the node in the ONNX model.
   */
  void addNodeAttribute(const std::string &attributeName,
                        const std::string &attributeValue,
                        const std::set<TensorId> &nodeOutputNames);

  /**
   * Add an attribute to the ONNX node which is uniquely identified by the
   * output tensors.
   *
   * This function will throw an exception if it cannot find the unique
   * node or if the attribute already exists.
   *
   * \param attributeName The name of the attribute to add.
   * \param attributeValue A \c char value of the attribute to add.
   * \param nodeOutputNames The tensor ids of the output tensors of the
   *     ONNX node used to find the node in the ONNX model.
   */
  void addNodeAttribute(const std::string &attributeName,
                        const char *attributeValue,
                        const std::set<TensorId> &nodeOutputNames);

  /**
   * Add an attribute to the ONNX node which is uniquely identified by the
   * output tensors.
   *
   * This function will throw an exception if it cannot find the unique
   * node or if the attribute already exists.
   *
   * \param attributeName The name of the attribute to add.
   * \param attributeValue A \c std::vector<std::string> value of the attribute
   *      to add.
   * \param nodeOutputNames The tensor ids of the output tensors of the
   *     ONNX node used to find the node in the ONNX model.
   */
  void addNodeAttribute(const std::string &attributeName,
                        const std::vector<std::string> &attributeValue,
                        const std::set<TensorId> &nodeOutputNames);

  /**
   * Add an attribute to the ONNX node which is uniquely identified by the
   * output tensors.
   *
   * This function will throw an exception if it cannot find the unique
   * node or if the attribute already exists.
   *
   * \param attributeName The name of the attribute to add.
   * \param attributeValue A bool value of the attribute to add.
   * \param nodeOutputNames The tensor ids of the output tensors of the
   *     ONNX node used to find the node in the ONNX model.
   */
  void addNodeAttribute(const std::string &attributeName,
                        const bool attributeValue,
                        const std::set<TensorId> &nodeOutputNames);

  /**
   * Add an attribute to the ONNX node which is uniquely identified by the
   * output tensors.
   *
   * This function will throw an exception if it cannot find the unique
   * node or if the attribute already exists.
   *
   * \param attributeName The name of the attribute to add.
   * \param attributeValue A constant tensor initializer.
   * \param nodeOutputNames The tensor ids of the output tensors of the
   *     ONNX node used to find the node in the ONNX model.
   */
  void addNodeAttribute(const std::string &attributeName,
                        const ConstVoidData &attributeValue,
                        const std::set<TensorId> &nodeOutputNames);

  /**
   * Check whether the ONNX node has an attribute set.
   *
   * This function will throw an exception if it cannot find the unique
   * node.
   *
   * \param attributeName The name of the attribute to find.
   * \param nodeOutputNames The tensor ids of the output tensors of the
   *     ONNX node used to find the node in the ONNX model.
   * \return `true` if the node has an attribute set; `false` otherwise.
   */
  bool nodeHasAttribute(const std::string &attributeName,
                        const std::set<TensorId> &nodeOutputNames);

  /**
   * Get the value of an attribute for the ONNX node where the value is a
   * \c int64_t.
   *
   * This function will throw an exception if it cannot find the unique
   * node or if the attribute does not exist or if it has not been set to the
   * \c int64_t type.
   *
   * \param attributeName The name of the attribute to find.
   * \param nodeOutputNames The tensor ids of the output tensors of the
   *     ONNX node used to find the node in the ONNX model.
   * \return Value of the attribute.
   */
  int64_t getInt64NodeAttribute(const std::string &attributeName,
                                const std::set<TensorId> &nodeOutputNames);

  /**
   * Get the value of an attribute for the ONNX node where the value is a
   * \c std::vector<int64_t>.
   *
   * This function will throw an exception if it cannot find the unique
   * node or if the attribute does not exist or if it has not been set to the
   * \c std::vector<int64_t> type.
   *
   * \param attributeName The name of the attribute to find.
   * \param nodeOutputNames The tensor ids of the output tensors of the
   *     ONNX node used to find the node in the ONNX model.
   * \return Value of the attribute.
   */
  std::vector<int64_t>
  getInt64VectorNodeAttribute(const std::string &attributeName,
                              const std::set<TensorId> &nodeOutputNames);

  /**
   * Get the value of an attribute for the ONNX node where the value is a
   * \c float.
   *
   * This function will throw an exception if it cannot find the unique
   * node or if the attribute does not exist or if it has not been set to the
   * \c float type.
   *
   * \param attributeName The name of the attribute to find.
   * \param nodeOutputNames The tensor ids of the output tensors of the
   *     ONNX node used to find the node in the ONNX model.
   * \return Value of the attribute.
   */
  float getFloatNodeAttribute(const std::string &attributeName,
                              const std::set<TensorId> &nodeOutputNames);

  /**
   * Get the value of an attribute for the ONNX node where the value is a \c
   * std::vector<float>.
   *
   * This function will throw an exception if it cannot find
   * the unique node or if the attribute does not exist.
   *
   * \param attributeName The name of the attribute to find.
   * \param nodeOutputNames The tensor ids of the output tensors of the
   *     ONNX node used to find the node in the ONNX model.
   * \return Value of the attribute.
   */
  std::vector<float>
  getFloatVectorNodeAttribute(const std::string &attributeName,
                              const std::set<TensorId> &nodeOutputNames);

  /**
   * Get the value of an attribute for the ONNX node where the value is a
   * string.
   *
   * This function will throw an exception if it cannot find the unique
   * node or the attribute does not exist or it has not been set to the
   * \c std::string type.
   *
   * \param attributeName The name of the attribute for which the value is
   *     required.
   * \param nodeOutputNames The tensor ids of the output tensors of the
   *      ONNX node used to find the node in the ONNX model.
   * \return Value of the attribute.
   */
  std::string getStringNodeAttribute(const std::string &attributeName,
                                     const std::set<TensorId> &nodeOutputNames);

  /**
   * Get the value of an attribute for the ONNX node where the value is a vector
   * of strings.
   *
   * This function will throw an exception if it cannot find the unique
   * node or if the attribute does not exist.
   *
   * \param attributeName The name of the attribute for which the value is
   *     required.
   * \param nodeOutputNames The tensor ids of the output tensors of the
   *     ONNX node used to find the node in the ONNX model.
   * \return Value of the attribute.
   */
  std::vector<std::string>
  getStringVectorNodeAttribute(const std::string &attributeName,
                               const std::set<TensorId> &nodeOutputNames);

  /**
   * Get the value of an attribute for the ONNX node where the value is a
   * boolean.
   *
   * This function will throw an exception if it cannot find the unique node or
   * if the attribute does not exist.
   *
   * \param attributeName The name of the attribute for which the value is
   *     required.
   * \param nodeOutputNames The tensor ids of the output tensors of the ONNX
   *     node used to find the node in the ONNX model.
   * \return Value of the attribute.
   */
  bool getBoolNodeAttribute(const std::string &attributeName,
                            const std::set<TensorId> &nodeOutputNames);

  /**
   * Remove an attribute from the ONNX node.
   * This function will throw an exception if it cannot find the unique
   * node or if the attribute does not exist.
   *
   * \param attributeName The name of the attribute to find.
   * \param nodeOutputNames The tensor ids of the output tensors of the ONNX
   *     node used to find the node in the ONNX model.
   */
  void removeNodeAttribute(const std::string &attributeName,
                           const std::set<TensorId> &nodeOutputNames);

  /**
   * Get all the attribute names from the ONNX node.
   * This function will throw an exception if it cannot find the unique
   * node.
   *
   * \param nodeOutputNames The tensor ids of the output tensors of the ONNX
   *      node used to find the node in the ONNX model.
   * \return The attribute names associated with the ONNX node.
   */
  std::vector<std::string>
  getAllNodeAttributeNames(const std::set<TensorId> &nodeOutputNames);

  /**
   * Get the index of the virtual graph that computes this node.
   * This applies in a multi IPU system.
   *
   * This function will throw an exception if the virtual graph has not been set
   * in the current scope.
   *
   * \param nodeOutputName The tensor id of the output tensor of the ONNX node
   *      used to find the node in the ONNX model.
   * \return The virtual graph associated with the ONNX node.
   */
  int64_t getVirtualGraph(const TensorId &nodeOutputName) {
    return getInt64NodeAttribute(sVirtualGraphAttribute, {nodeOutputName});
  }

  /**
   * Get the index of the virtual graph that computes this node based on
   * multiple output tensors. This applies in a multi IPU system.
   *
   * This function will throw an exception if the virtual graph has not been set
   * in the current scope.
   *
   * \param nodeOutputNames The tensor ids of the output tensors of the ONNX
   *      node used to find the node in the ONNX model.
   * \return The virtual graph associated with the ONNX node.
   */
  int64_t getVirtualGraph(const std::set<TensorId> &nodeOutputNames) {
    return getInt64NodeAttribute(sVirtualGraphAttribute, nodeOutputNames);
  }

  /**
   * Get the execution phase for a single output tensor.
   * This only applies to a multi-IPU system.
   *
   * This function will throw an exception if the execution phase has not been
   * set in the current scope.
   *
   * \param nodeOutputNames The tensor id of the output tensor of the ONNX node
   *      used to find the node in the ONNX model.
   * \return The execution phase associated with the ONNX node.
   */
  int64_t getExecutionPhase(const TensorId &nodeOutputName) {
    return getInt64NodeAttribute(sExecutionPhaseAttribute, {nodeOutputName});
  }

  /**
   * Get the execution phase for a set of output tensors.
   * This only applies to a multi-IPU system.
   *
   * This function will throw an exception if the execution phase has not been
   * set in the current scope.
   *
   * \param nodeOutputNames The tensor ids of the output tensors of the ONNX
   *      node used to find the node in the ONNX model.
   * \return The execution phase associated with the ONNX node.
   */
  int64_t getExecutionPhase(const std::set<TensorId> &nodeOutputNames) {
    return getInt64NodeAttribute(sExecutionPhaseAttribute, nodeOutputNames);
  }

  /**
   * Retrieve the ONNX serialized ModelProto.
   *
   * \param humanReadable If true, return a human readable text representation
   *                      of the model, otherwise use a binary format.
   * \return A serialized ONNX ModelProto.
   */
  std::string getModelProto(bool humanReadable = false) const;

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
   * \param ids The names of tensors for which data is to be saved externally.
   * \param fn The name of a file containing the binary tensor data. This
   *     can be an absolute or relative path. If a relative path, when
   *     the ONNX model is saved, external tensor data will be written
   *     to a path relative to the current working directory.
   */
  void saveInitializersExternally(const std::vector<TensorId> &ids,
                                  const std::string &fn);

  /**
   * Return a list of ONNX graph input tensor ids.
   *
   * \return A vector of input tensor ids.
   */
  std::vector<TensorId> getInputTensorIds() const;

  /**
   * Return a list of ONNX graph output tensor ids.
   *
   * \return A vector of output tensor ids.
   */
  std::vector<TensorId> getOutputTensorIds() const;

  /**
   * Return a list of ONNX graph value tensor ids.
   *
   * These tensors are stored in the `value_info` section
   * of the ONNX `GraphProto` structure.
   *
   * \return A vector of value tensor names.
   */
  std::vector<TensorId> getValueTensorIds() const;

  /**
   * Return a list of ONNX graph initialized tensor ids.
   *
   * These tensors are stored in the `initialized` section of the ONNX
   * `GraphProto` structure..
   *
   * \return A vector of names of initialized tensors.
   */
  std::vector<TensorId> getTrainableTensorIds() const;

  /**
   * Check if a tensor has value info.
   *
   * A tensor may not have value info if this either does not exist or if
   * shape inference has failed.
   *
   * \return `True` if the tensor has value info; `false` otherwise..
   */
  bool hasValueInfo(const TensorId &id) const;

  /**
   * Return an ONNX graph tensor shape, from either the `input`,
   * `output`, or `value_info` lists in `GraphProto`.
   *
   * \param id The id of the tensor for which dimensions are required.
   * \return A vector of the tensor dimensions.
   */
  std::vector<int64_t> getTensorShape(const TensorId id);

  /**
   * Check if the ONNX tensor is in the initializer list
   * of `GraphProto`.
   *
   * \param id A tensor id.
   * \return `True` if the tensor is in the initializer list; `false` otherwise.
   */
  bool isInitializer(const TensorId id) const;

  /**
   * Return an ONNX graph tensor type as a lower case string, from either
   * the `input`, `output`, or `value_info` lists in `GraphProto`.
   *
   * \param id The id of the tensor for which the type is required.
   * \return A lower case string of the tensor data type.
   */
  std::string getTensorDtypeString(const TensorId id);

  /**
   * Return a tensor type from either
   * the `input`, `output`, or `value_info` lists in `GraphProto`.
   *
   * \param id The id of tensor id for which the type is required.
   * \return The data type of the tensor.
   */
  DataType getTensorDataType(const TensorId id);

  /**
   * Push a name onto the name scope stack.
   *
   * The names of tensors and nodes added to the ONNX graph will be prefixed
   * with a concatenation of the names in the name scope stack.
   * \param name The tensor name to be pushed onto the name scope stack.
   */
  void pushNameScope(const std::string &name);

  /**
   * Remove the last entry in the name scope stack.
   */
  void popNameScope();

  /**
   * Get the current name scope stack using the default delimiter.
   *
   * \param name (Optional) A string to concatenate to the end of the stack.
   * \return A string of the concatenated name scope stack.
   */
  std::string getNameScope(const std::string &name = "") const;

  /**
   * Set a graph name.
   *
   * \param name The string to name the graph.
   */
  void setGraphName(const std::string &name);

  /**
   * Set the parent graph of this builder.
   *
   * \param parent The builder to set as the parent of this builder.
   */
  void setParent(Builder *parent);

  /**
   * Return the parent graph of this builder or null if there is no parent.
   */
  Builder *getParent() const;

  /**
   * Check if this builder represents a subgraph.
   *
   * \returns If `true` then the builder represents a subgraph. If `false` then
   *      the builder does not represent a subgraph.
   */
  bool hasParent() const { return parent == nullptr; }

  /**
   * Embed the value of replicationFactor into the OnnxModel.
   * Should be interpreted as 1 if not present in the model.
   * \param replicationFactor The replication factor.
   */
  void embedReplicationFactor(int replicationFactor);

private:
  void configure();
  void configure(const std::string &modelProtoOrFilename);

  /**
   * Load a serialized ONNX ModelProto into the builder and validate it.
   *
   * \param modelProtoOrFilename An ONNX model protobuf, or the name of a
   *     file containing an ONNX model protobuf.
   */
  void loadModelProto(const std::string &modelProtoOrFilename);

  std::unique_ptr<BuilderImpl> impl_;
  std::map<int, std::unique_ptr<Builder>> children;
  int nChildren{0};

  Builder *parent;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_BUILDER_HPP_
