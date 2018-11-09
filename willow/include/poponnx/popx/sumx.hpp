#ifndef GUARD_NEURALNET_SUMX_HPP
#define GUARD_NEURALNET_SUMX_HPP

#include <poponnx/names.hpp>
#include <poponnx/popx/opx.hpp>

namespace willow {

class SumOp;

namespace popx {

class SumOpx : public Opx {
public:
  SumOpx(Op *, Devicex *);
  SumOp *getSumOp() const;
  void grow() const override final;
};

} // namespace popx
} // namespace willow

#endif
