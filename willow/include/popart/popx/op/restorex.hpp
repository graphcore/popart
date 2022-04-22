// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_RESTOREX_HPP
#define GUARD_NEURALNET_RESTOREX_HPP

#include <cstdint>
#include <snap/Tensor.hpp>
#include <popart/popx/popopx.hpp>

namespace snap {
namespace program {
class Sequence;
} // namespace program
} // namespace snap

namespace popart {

class RestoreOp;
class RestoreInplaceOp;
class Op;

namespace popx {
class Devicex;

/**
 * \brief Base class for restore opxs.
 *
 * \tparam Opx is subclass of RestoreBaseOpx. Must have type alias `OpType`
 * defined as the Op that it corresponds to.
 */
template <typename Derived> class RestoreBaseOpx : public PopOpx {
public:
  RestoreBaseOpx(Op *op, Devicex *devicex);

  void grow(snap::program::Sequence &) const = 0;

protected:
  bool canDynamicSliceRestore;

  snap::Tensor growRestore(snap::program::Sequence &prog,
                           const snap::Tensor &stash) const;

private:
  snap::Tensor growStaticSliceRestore(snap::program::Sequence &prog,
                                      int64_t stashSize,
                                      const snap::Tensor &stashIndex,
                                      const snap::Tensor &stash) const;
  snap::Tensor growDynamicSliceRestore(snap::program::Sequence &prog,
                                       const snap::Tensor &stashIndex,
                                       const snap::Tensor &stash) const;
};

class RestoreOpx final : public RestoreBaseOpx<RestoreOpx> {
public:
  RestoreOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;

  using OpType = RestoreOp;
};

class RestoreInplaceOpx final : public RestoreBaseOpx<RestoreInplaceOpx> {
public:
  RestoreInplaceOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;

  using OpType = RestoreInplaceOp;
};

} // namespace popx
} // namespace popart

#endif
