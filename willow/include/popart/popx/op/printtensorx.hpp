// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_PRINTTENSORX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_PRINTTENSORX_HPP_

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

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_PRINTTENSORX_HPP_
