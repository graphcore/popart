// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_SOFTMAXX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_SOFTMAXX_HPP_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <poplar/Tensor.hpp>
#include <popart/popx/op/elementwisex.hpp>

#include "popart/popx/debugcontextx.hpp"
#include "popart/popx/opx.hpp"

namespace poplar {
class Graph;

namespace program {
class Sequence;
} // namespace program
} // namespace poplar

namespace popart {
class Op;

namespace popx {
class Devicex;

class SoftmaxComputex : public EwuComputex {

public:
  SoftmaxComputex(int64_t ax, bool ens, const std::vector<size_t> &os)
      : axis(ax), enableNonStable(ens), outShape(os) {}

  poplar::Tensor outplace(poplar::program::Sequence &,
                          poplar::Graph &,
                          const poplar::Tensor &,
                          const poplar::DebugNameAndId &,
                          const std::string &) const final;

  void inplace(poplar::program::Sequence &,
               poplar::Graph &,
               const poplar::Tensor &,
               const poplar::DebugNameAndId &,
               const std::string &) const final;

  static std::unique_ptr<EwuComputex>
  get(int64_t axis, bool ens, const std::vector<size_t> &os) {
    return std::unique_ptr<EwuComputex>(new SoftmaxComputex(axis, ens, os));
  }

  poplar::Tensor reshape(const poplar::Tensor &) const final;

  void setAxis(int64_t a) { axis = a; }

private:
  int64_t axis;
  bool enableNonStable;
  std::vector<size_t> outShape;
};

class SoftmaxOpx : public ElementWiseUnaryOutplaceOpx {
public:
  SoftmaxOpx(Op *, Devicex *);
};

class SoftmaxInplaceOpx : public ElementWiseUnaryInplaceOpx {
public:
  SoftmaxInplaceOpx(Op *, Devicex *);
};

// compute dL/dv from v and dp, where p = softmax(v)
class SoftmaxGradOpx : public ElementWiseUnaryOpx {
public:
  SoftmaxGradOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

// compute dL/dv from lab and p, where p = softmax(v), L = nll(p, lab)
class SoftmaxGradDirectOpx : public Opx {
public:
  SoftmaxGradDirectOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

// As above, but combines the loss calculation to reduce redundancy
class NlllWithSoftmaxGradDirectOpx : public Opx {
public:
  NlllWithSoftmaxGradDirectOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;

private:
  void handleLossOutNotReducedToScalar(poplar::Tensor &reduction,
                                       const poplar::Tensor &label,
                                       poplar::Tensor &label1D,
                                       poplar::program::Sequence &prog) const;
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_SOFTMAXX_HPP_
