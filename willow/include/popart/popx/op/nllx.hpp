// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_NLLX_HPP
#define GUARD_NEURALNET_NLLX_HPP

#include <popart/names.hpp>
#include <popart/op/identity.hpp>
#include <popart/popx/opx.hpp>

namespace popart {
namespace popx {

class NllOpx : public Opx {
public:
  NllOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
  // Mask the loss, or loss-grad of rows (i.e. samples) of tensor t
  // whose corresponding target label is equal to ignoreIndex

  static void flattenAndEncodeOneHot(const Opx &opx,
                                     poplar::program::Sequence &prog,
                                     const poplar::Tensor &probs,
                                     const poplar::Tensor &label,
                                     poplar::Tensor &probs2D,
                                     poplar::Tensor &label1D,
                                     poplar::Tensor &oneHot);

  static poplar::Tensor
  applyMaskInPlaceForIgnoredIndex(const Opx &opx,
                                  poplar::Tensor t,
                                  poplar::Tensor labels,
                                  int ignoreIndex,
                                  poplar::program::Sequence &prog);
  // If the loss that created this op was constructed with a
  // ReductionType 'Mean', then we scale the output of the loss
  // tensor by 1/local_loss_elements and the gradient of the loss tensor
  // by 1/(local_loss_elements * replication)
  // This is a static function that is used to scale the every Nll
  // loss and loss grad at the output of the respective ops/grad ops
  static void applyScalingInPlaceForMeanReduction(
      const Opx &opx,
      poplar::Tensor t,
      poplar::Tensor scale,
      poplar::program::Sequence &prog,
      const ScaleByReplication scaleByReplication = ScaleByReplication::Yes);

  // Same as above, except the divisor for the scaling of the loss/
  // loss grad cannot be determined at compile time.
  // If the user has specified an ignoreIndex for the loss, then
  // samples with a label corresponding to this index should not
  // be taken into account when scaling the output.
  // The mask tensor generated in applyMaskInPlaceForIgnoredIndex
  // is used to dynamically determine the aprropriate scale factor
  static void applyScalingInPlaceForMeanReductionWithIgnoreIndex(
      const Opx &opx,
      poplar::Tensor t,
      poplar::Tensor scale,
      poplar::Tensor mask,
      poplar::program::Sequence &prog,
      const ScaleByReplication scaleByReplication = ScaleByReplication::Yes);

  static void handleLossGradScaling(const Opx &opx,
                                    bool hasIgnoreIndex,
                                    int64_t ignoreIndex,
                                    bool meanReduce,
                                    poplar::Tensor &oneHot,
                                    poplar::Tensor &gradIn,
                                    poplar::Tensor &label1D,
                                    poplar::program::Sequence &prog);

  static void handleLossOutReducedToScalar(const Opx &opx,
                                           bool hasIgnoreIndex,
                                           int64_t ignoreIndex,
                                           bool meanReduce,
                                           poplar::Tensor &reduction,
                                           poplar::Tensor &label1D,
                                           poplar::program::Sequence &prog,
                                           const OutIndex outIdx);

  static void handleLossOutNotReducedToScalar(const Opx &opx,
                                              poplar::Tensor &reduction,
                                              const poplar::Tensor &label,
                                              poplar::Tensor &label1D,
                                              poplar::program::Sequence &prog);
};

class NllGradOpx : public Opx {
public:
  NllGradOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif
