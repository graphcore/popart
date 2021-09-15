// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_PRINTTENSORX_HPP
#define GUARD_NEURALNET_PRINTTENSORX_HPP

#include <popart/names.hpp>
#include <popart/popx/popopx.hpp>

namespace popart {

namespace popx {

class PrintTensorOpx : public PopOpx {
public:
  PrintTensorOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;

private:
  std::string getTitle() const;
};

} // namespace popx
} // namespace popart

#endif
