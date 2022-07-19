// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_CLIPX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_CLIPX_HPP_

#include "popart/popx/debugcontextx.hpp"
#include <memory>
#include <snap/Tensor.hpp>
#include <string>
#include <poplar/Type.hpp>
#include <popart/popx/op/elementwisex.hpp>

#include "popart/popx/popopx.hpp"

namespace snap {
class Graph;
namespace program {
class Sequence;
} // namespace program
} // namespace snap

namespace popart {
class ClipInplaceOp;
class ClipOp;
class Op;

namespace popx {
class Devicex;

class ClipComputex : public EwuComputex {
public:
  ClipComputex(float min_, float max_) : min(min_), max(max_) {}

  snap::Tensor outplace(snap::program::Sequence &,
                        snap::Graph &,
                        const snap::Tensor &tensor,
                        const poplar::DebugNameAndId &,
                        const std::string &) const final;

  void inplace(snap::program::Sequence &,
               snap::Graph &,
               const snap::Tensor &,
               const poplar::DebugNameAndId &,
               const std::string &) const final;

  static std::unique_ptr<EwuComputex> get(float min, float max) {
    return std::unique_ptr<EwuComputex>(new ClipComputex(min, max));
  }

  static snap::Tensor getClipTensor(float val,
                                    const poplar::Type &type,
                                    snap::Graph &graph,
                                    const poplar::DebugNameAndId &);
  static snap::Tensor broadcastClipTensor(snap::Tensor clipT,
                                          const snap::Tensor &refT);

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

class ClipGradOpx : public PopOpx {
public:
  ClipGradOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_CLIPX_HPP_
