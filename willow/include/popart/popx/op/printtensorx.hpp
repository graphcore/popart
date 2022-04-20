// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_PRINTTENSORX_HPP
#define GUARD_NEURALNET_PRINTTENSORX_HPP

#include <string>
#include <popart/popx/popopx.hpp>

namespace snap {
namespace program {
class Sequence;
} // namespace program
} // namespace snap

namespace popart {
class Op;

namespace popx {
class Devicex;

class PrintTensorOpx : public PopOpx {
public:
  PrintTensorOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;

private:
  std::string getTitle() const;
};

} // namespace popx
} // namespace popart

#endif
