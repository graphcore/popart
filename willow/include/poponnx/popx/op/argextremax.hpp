#ifndef GUARD_NEURALNET_ARGEXTREMAX_HPP
#define GUARD_NEURALNET_ARGEXTREMAX_HPP

#include <poponnx/names.hpp>
#include <poponnx/popx/opx.hpp>

namespace poponnx {

namespace popx {

class ArgExtremaOpx : public Opx {
public:
  ArgExtremaOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const override;

private:
  virtual poplar::Tensor extremaOp(poplar::program::Sequence &,
                                   const poplar::Tensor &) const = 0;
};

} // namespace popx
} // namespace poponnx

#endif
