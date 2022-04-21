// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_NORMX_HPP
#define GUARD_NEURALNET_NORMX_HPP

#include <popart/names.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/normx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/popx/popopx.hpp>

#include <snap/Tensor.hpp>

// Base class for the norm options such as group, instance, batch

namespace poplar {
using Shape = std::vector<std::size_t>;
}

namespace popart {
namespace popx {

class NormOpx : public PopOpx {
public:
  NormOpx(Op *, Devicex *);

protected:
  snap::Tensor convertInvSdToVar(snap::program::Sequence &prog,
                                 const snap::Tensor &invSd,
                                 float epsilon,
                                 const poplar::Type dstType) const;

  snap::Tensor convertVarToInvSd(snap::program::Sequence &prog,
                                 const snap::Tensor &var,
                                 float epsilon,
                                 const poplar::Type dstType) const;

private:
};

} // namespace popx
} // namespace popart

#endif
