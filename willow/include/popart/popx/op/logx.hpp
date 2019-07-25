#ifndef GUARD_NEURALNET_LOGX_HPP
#define GUARD_NEURALNET_LOGX_HPP

#include <popart/names.hpp>
#include <popart/popx/op/elementwisex.hpp>

namespace popart {

namespace popx {

class LogOpx : public ElementWiseUnaryOpx {
public:
  LogOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif
