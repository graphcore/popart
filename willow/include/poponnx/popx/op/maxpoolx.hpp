#ifndef GUARD_NEURALNET_MAXPOOLX_HPP
#define GUARD_NEURALNET_MAXPOOLX_HPP

#include <poponnx/names.hpp>
#include <poponnx/popx/opx.hpp>

namespace willow {

class MaxPoolOp;
class MaxPoolGradOp;

namespace popx {

class MaxPoolOpx : public Opx {
public:
  MaxPoolOpx(Op *, Devicex *);
  MaxPoolOp *getMaxPoolOp() const;
  void grow(poplar::program::Sequence &) const final;
};

class MaxPoolGradOpx : public Opx {
public:
  MaxPoolGradOpx(Op *, Devicex *);
  MaxPoolGradOp *getMaxPoolGradOp() const;
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace willow

#endif
