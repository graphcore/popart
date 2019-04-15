#ifndef GUARD_NEURALNET_TOPKX_HPP
#define GUARD_NEURALNET_TOPKX_HPP

#include <poponnx/names.hpp>
#include <poponnx/popx/op/basesortx.hpp>

namespace poponnx {

namespace popx {

class TopKOpx : public BaseSortOpx {
public:
  TopKOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;

private:
  unsigned K;
};

} // namespace popx
} // namespace poponnx

#endif
