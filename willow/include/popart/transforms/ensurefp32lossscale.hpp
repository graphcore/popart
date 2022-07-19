// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_TRANSFORMS_ENSUREFP32LOSSSCALE_HPP_
#define POPART_WILLOW_INCLUDE_POPART_TRANSFORMS_ENSUREFP32LOSSSCALE_HPP_

#include <cstddef>
#include <string>
#include <utility>
#include <vector>
#include <popart/transforms/transform.hpp>

#include "popart/names.hpp"

namespace popart {
class Graph;
class Op;
class Tensor;

// This transform can be applied to graphs with an fp16 loss scale tensor.
// It aims to keep the loss scale tensor in fp32 until as late as possible -
// that is, until it is combined with an fp16 tensor by some operation in the
// backwards pass. This enables the model to support much larges loss scale
// values than max(fp16).
//
// Some background context:
// In PopART's autograd transform, if the TrainingSession is constructed with
// a loss TensorId corresponding to an fp16 tensor, and an optimizer with loss
// scaling, then an fp16 loss scale tensor is created in the the IR, and passed
// as an input to the grad op of the op that produces the final loss tensor.
//
// Motivation for this transform:
// We want to be able to support loss scale values > max(fp16), even in the case
// when the loss gradient op prodcues an fp16 output. A large loss scale may
// be set by the user, or as the result of updates by the automatic loss scaling
// algorithm, if automatic loss scaling is enabled.
//
// The transform:
//
// The main component is a graph traversal of the backwards pass, starting from
// the loss scale tensor:
//   - We 'pass through' single-input ops that do not combine the loss scale
//     with an activation tensor.
//   - Otherwise we terminate the traversal. We refer to these terminal ops
//     as 'mixed precision loss grad op' (or MPLGO) candidates
//
// To apply the transform, we convert the loss scale tensor from fp16 to fp32,
// and all pass-through ops are converted to fp32-input, fp32-output.
//
// So in the most basic case, the transform replaces:
//    activation_fp16 --.
//    lossScale_fp16 -- MPLGO -- grad_fp16
// with:
//    activation_fp16 --.
//    lossScale_fp32 -- MPLGO -- grad_fp16
//
// A more complex example. Replace:
//                                          act_fp16 --.
// lossScale_fp16 -- PassThroughOp0 -- grad0_fp16 -- MPLGO0 -- grad2_fp16
//         \
//          \                               act_fp16 --.
//           ------- PassThroughOp1 -- grad1_fp16 -- MPLGO1 -- grad3_fp16
// with:
//                                          act_fp16 --.
// lossScale_fp32 -- PassThroughOp0 -- grad0_fp32 -- MPLGO0 -- grad2_fp16
//         \
//          \                               act_fp16 --.
//           ------- PassThroughOp1 -- grad1_fp32 -- MPLGO1 -- grad3_fp16
//
// If an MPLGO candidate is not a valid MPLGO, its fp16 inputs are upcast
// so that the computation occurs in fp32, and its fp32 outputs are downcast,
// so that the consumers (and the rest of the backwards pass) are unaffected.
// Again, for the basic case:
//    act_fp16 -----------.
//    lossScale_fp16 -- Non-MPLGO -- grad_fp16
// is replaced with:
//    act_fp16 -- CastOp -- act_fp32 --.
//    lossScale_fp32 ------------ Non-MPLGO -- grad_fp32 -- CastOp -- grad_fp16
using PassThroughOps            = std::vector<Op *>;
using TerminalOps               = std::vector<Op *>;
using FromLossScaleTraversalOps = std::pair<PassThroughOps, TerminalOps>;

class EnsureFp32LossScale : public Transform {
public:
  static std::size_t id();

  EnsureFp32LossScale() : Transform() {}
  virtual ~EnsureFp32LossScale() override {}

  virtual bool apply(Graph &graph) const final;

  virtual std::size_t getId() const final { return id(); }

  virtual std::string getName() const final { return "EnsureFp32LossScale"; }

  /**
   * To return true, the op's implementation must be able to handle mixed
   * precision maths. We have no good way to know this programmatically at the
   * point of running this transform, so we hard code this information here.
   *
   * \param op The op we want to check if it has an impelemntation that is
   *     known to support mixed precision inputs.
   * \return True if it is has an implementation known to support mixed
   *     precision inputs.
   **/
  static bool isMixedPrecisionLossGradOp(Op *op);

  /**
   * Only to be called on an op for which a call to
   * \a isMixedPrecisionLossGradOp return true.
   *
   * \param op An MPLGO candidate whose loss scale tensor (or descendant
   *     there-of) you want to find.
   * \return The input tensor.
   **/
  static Tensor *getLossScaleInputTensor(Op *op);

  /**
   * For deciding whether to continue graph traversal from \a op's outputs,
   * or to terminate the traversal at this op.
   *
   * \param op The op.
   * \return True if the op has a single input, and all its outputs are of the
   *     same type as the input.
   **/
  bool isPassThroughOp(Op *op) const;

  /**
   * Traverse the graph from the loss scale tensor.
   *  - We 'pass through' single-input ops that do not combine the loss scale
   *    (or a descendant of it) with an activation tensor.
   *  - Otherwise we terminate the traversal. We refer to these terminal ops
   *    as 'mixed precision loss grad op' (or MPLGO) candidates.
   *
   * \param graph The graph to be traversed.
   * \return A pair containing the list of pass-through ops and MPLGO
   *     candidates.
   **/
  FromLossScaleTraversalOps
  traverseFromLossScaleTensor(const Graph &graph) const;

  /**
   * Run the checks to see if the transform should be applied.
   *
   * \param graph The graph that the checks are run on.
   * \return True if the checks pass.
   **/
  bool shouldApply(const Graph &graph) const;

  /**
   * Upcast fp16 tensor at input index \a index to \a op to fp32. This is done
   * by disconnecting the input tensor, inserting a CastOp, and re-connecting
   * the output tensor of the CastOp at \a index.
   *
   * \param op The op whose input is to be upcast.
   * \param index The input index to \a op at which the tensor is to be upcast.
   **/
  void upCastTensor(Op *op, InIndex index) const;

  /**
   * Downcast fp16 \a tensor to fp16. This is done by disconnecting it from
   * its consumers, inserting a CastOp, and re-connecting the output tensor
   * of the CastOp to the consumers.
   *
   * \param tensor The tensor to be downcast.
   **/
  void downCastTensor(Tensor *tensor) const;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_TRANSFORMS_ENSUREFP32LOSSSCALE_HPP_
