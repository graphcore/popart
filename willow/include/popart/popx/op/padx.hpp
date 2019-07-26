#ifndef GUARD_NEURALNET_PADX_HPP
#define GUARD_NEURALNET_PADX_HPP

#include <popart/names.hpp>
#include <popart/popx/opx.hpp>

namespace popart {

class PadOp;

namespace popx {

class PadOpx : public Opx {
public:
  PadOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;

private:
  PadOp *getPadOp() const;
};

class PadInplaceOpx : public Opx {
public:
  PadInplaceOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;

private:
  PadInplaceOp *getPadInplaceOp() const;
};

} // namespace popx
} // namespace popart

#endif
