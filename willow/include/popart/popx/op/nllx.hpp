// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_NLLX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_NLLX_HPP_

#include <cstdint>
#include <snap/Tensor.hpp>
#include <popart/names.hpp>
#include <popart/popx/popopx.hpp>

namespace snap {
namespace program {
class Sequence;
} // namespace program
} // namespace snap

namespace popart {
class Op;

namespace popx {
class Devicex;

class NllOpx : public PopOpx {
public:
  NllOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;
  // Mask the loss, or loss-grad of rows (i.e. samples) of tensor t
  // whose corresponding target label is equal to ignoreIndex

  static void flattenAndEncodeOneHot(const PopOpx &opx,
                                     snap::program::Sequence &prog,
                                     const snap::Tensor &probs,
                                     const snap::Tensor &label,
                                     snap::Tensor &probs2D,
                                     snap::Tensor &label1D,
                                     snap::Tensor &oneHot);

  static snap::Tensor
  applyMaskInPlaceForIgnoredIndex(const PopOpx &opx,
                                  snap::Tensor t,
                                  snap::Tensor labels,
                                  int ignoreIndex,
                                  snap::program::Sequence &prog);
  // If the loss that created this op was constructed with a
  // ReductionType 'Mean', then we scale the output of the loss
  // tensor by 1/local_loss_elements and the gradient of the loss tensor
  // by 1/(local_loss_elements * replication)
  // This is a static function that is used to scale the every Nll
  // loss and loss grad at the output of the respective ops/grad ops
  static void
  applyScalingInPlaceForMeanReduction(const PopOpx &opx,
                                      snap::Tensor t,
                                      snap::Tensor scale,
                                      snap::program::Sequence &prog);

  // Same as above, except the divisor for the scaling of the loss/
  // loss grad cannot be determined at compile time.
  // If the user has specified an ignoreIndex for the loss, then
  // samples with a label corresponding to this index should not
  // be taken into account when scaling the output.
  // The mask tensor generated in applyMaskInPlaceForIgnoredIndex
  // is used to dynamically determine the aprropriate scale factor
  static void applyScalingInPlaceForMeanReductionWithIgnoreIndex(
      const PopOpx &opx,
      snap::Tensor t,
      snap::Tensor scale,
      snap::Tensor mask,
      snap::program::Sequence &prog);

  static void handleLossGradScaling(const PopOpx &opx,
                                    bool hasIgnoreIndex,
                                    int64_t ignoreIndex,
                                    bool meanReduce,
                                    snap::Tensor &oneHot,
                                    snap::Tensor &gradIn,
                                    snap::Tensor &label1D,
                                    snap::program::Sequence &prog);

  static void handleLossOutReducedToScalar(const PopOpx &opx,
                                           bool hasIgnoreIndex,
                                           int64_t ignoreIndex,
                                           bool meanReduce,
                                           snap::Tensor &reduction,
                                           snap::Tensor &label1D,
                                           snap::program::Sequence &prog,
                                           const OutIndex outIdx);

  static void handleLossOutNotReducedToScalar(const PopOpx &opx,
                                              snap::Tensor &reduction,
                                              const snap::Tensor &label,
                                              snap::Tensor &label1D,
                                              snap::program::Sequence &prog);
};

class NllGradOpx : public PopOpx {
public:
  NllGradOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_NLLX_HPP_
