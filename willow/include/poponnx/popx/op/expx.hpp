#ifndef GUARD_NEURALNET_EXPX_HPP
#define GUARD_NEURALNET_EXPX_HPP

#include <poponnx/names.hpp>
#include <poponnx/popx/opx.hpp>

namespace poponnx {

namespace popx {

class ExpOpx : public Opx {
public:
  ExpOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace poponnx

#endif
