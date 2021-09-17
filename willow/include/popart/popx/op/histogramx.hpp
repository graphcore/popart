// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_HISTOGRAMX_HPP
#define GUARD_NEURALNET_HISTOGRAMX_HPP

#include <popart/popx/popopx.hpp>

namespace popart {
namespace popx {

class HistogramOpx : public PopOpx {
public:
  HistogramOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif
