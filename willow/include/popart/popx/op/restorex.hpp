// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_RESTOREX_HPP
#define GUARD_NEURALNET_RESTOREX_HPP

#include <popart/popx/popopx.hpp>

namespace popart {

class RestoreOp;
class RestoreOpx;

namespace popx {

/**
 * \brief Base class for restore opxs.
 *
 * \tparam Opx is subclass of RestoreBaseOpx. Must have type alias `OpType`
 * defined as the Op that it corresponds to.
 */
template <typename Derived> class RestoreBaseOpx : public PopOpx {
public:
  RestoreBaseOpx(Op *op, Devicex *devicex);

  void grow(poplar::program::Sequence &) const = 0;

protected:
  bool canDynamicSliceRestore;

  poplar::Tensor growRestore(poplar::program::Sequence &prog,
                             const poplar::Tensor &stash) const;

private:
  poplar::Tensor growStaticSliceRestore(poplar::program::Sequence &prog,
                                        int64_t stashSize,
                                        const poplar::Tensor &stashIndex,
                                        const poplar::Tensor &stash) const;
  poplar::Tensor growDynamicSliceRestore(poplar::program::Sequence &prog,
                                         const poplar::Tensor &stashIndex,
                                         const poplar::Tensor &stash) const;
};

class RestoreOpx final : public RestoreBaseOpx<RestoreOpx> {
public:
  RestoreOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;

  using OpType = RestoreOp;
};

class RestoreInplaceOpx final : public RestoreBaseOpx<RestoreInplaceOpx> {
public:
  RestoreInplaceOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;

  using OpType = RestoreInplaceOp;
};

} // namespace popx
} // namespace popart

#endif
