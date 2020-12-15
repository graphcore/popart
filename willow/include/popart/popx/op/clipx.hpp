// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_CLIPX_HPP
#define GUARD_NEURALNET_CLIPX_HPP

#include <popart/names.hpp>
#include <popart/popx/op/elementwisex.hpp>

namespace popart {

namespace popx {

class ClipComputex : public EwuComputex {
public:
  ClipComputex(float min_, float max_) : min(min_), max(max_) {}

  poplar::Tensor outplace(poplar::program::Sequence &,
                          poplar::Graph &,
                          const poplar::Tensor &tensor,
                          const poplar::DebugNameAndId &,
                          const std::string &) const final;

  void inplace(poplar::program::Sequence &,
               poplar::Graph &,
               const poplar::Tensor &,
               const poplar::DebugNameAndId &,
               const std::string &) const final;

  static std::unique_ptr<EwuComputex> get(float min, float max) {
    return std::unique_ptr<EwuComputex>(new ClipComputex(min, max));
  }

  static poplar::Tensor getClipTensor(float val,
                                      const poplar::Type &type,
                                      poplar::Graph &graph,
                                      const poplar::DebugNameAndId &);
  poplar::Tensor broadcastClipTensor(poplar::Tensor clipT,
                                     const poplar::Tensor &refT) const;

  static float getMinFromClipOp(Op *op);
  static float getMaxFromClipOp(Op *op);
  static float getMinFromClipInplaceOp(Op *op);
  static float getMaxFromClipInplaceOp(Op *op);

private:
  static ClipOp *getClipOpFromOp(Op *op);
  static ClipInplaceOp *getClipInplaceOpFromOp(Op *op);
  float min;
  float max;
};

class ClipOpx : public ElementWiseUnaryOutplaceOpx {
public:
  ClipOpx(Op *, Devicex *);
};

class ClipInplaceOpx : public ElementWiseUnaryInplaceOpx {
public:
  ClipInplaceOpx(Op *, Devicex *);
};

class ClipGradOpx : public Opx {
public:
  ClipGradOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif
