#ifndef GUARD_NEURALNET_LOGX_HPP
#define GUARD_NEURALNET_LOGX_HPP

#include <poponnx/names.hpp>
#include <poponnx/popx/opx.hpp>

namespace poponnx {

namespace popx {

class LogOpx : public Opx {
public:
  LogOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace poponnx

#endif
