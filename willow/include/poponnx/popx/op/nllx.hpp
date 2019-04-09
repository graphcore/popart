#ifndef GUARD_NEURALNET_NLLX_HPP
#define GUARD_NEURALNET_NLLX_HPP

#include <poponnx/names.hpp>
#include <poponnx/popx/opx.hpp>

namespace poponnx {

class NllGradOp;
class NllOp;

namespace popx {

class NllOpx : public Opx {
public:
  NllOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
  // Mask the loss, or loss-grad of rows (i.e. samples) of tensor t
  // whose corresponding target label is equal to ignoreIndex
  static void applyMaskInPlaceForIgnoredIndex(const Opx &opx,
                                              poplar::Graph &graph,
                                              poplar::Tensor t,
                                              poplar::Tensor labels,
                                              int ignoreIndex,
                                              poplar::program::Sequence &prog);
};

class NllGradOpx : public Opx {
public:
  NllGradOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace poponnx

#endif
