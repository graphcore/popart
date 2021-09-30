// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_INCREMENTMODX_HPP
#define GUARD_NEURALNET_INCREMENTMODX_HPP

#include <popart/names.hpp>
#include <popart/popx/op/elementwisex.hpp>

namespace popart {
namespace popx {

template <typename T> class IncrementModComputex : public EwuComputex {

public:
  IncrementModComputex(const Op *op);

  snap::Tensor outplace(snap::program::Sequence &,
                        snap::Graph &,
                        const snap::Tensor &,
                        const poplar::DebugNameAndId &,
                        const std::string &) const final;

  void inplace(snap::program::Sequence &,
               snap::Graph &,
               const snap::Tensor &,
               const poplar::DebugNameAndId &,
               const std::string &) const final;

  static std::unique_ptr<EwuComputex> get(const Op *op) {
    return std::unique_ptr<EwuComputex>(new IncrementModComputex<T>(op));
  }

public:
  const Op *op;
  T increment;
  T modulus;
};

class IncrementModOpx : public ElementWiseUnaryOutplaceOpx {
public:
  IncrementModOpx(Op *, Devicex *);
};

class IncrementModInplaceOpx : public ElementWiseUnaryInplaceOpx {
public:
  IncrementModInplaceOpx(Op *, Devicex *);
};

} // namespace popx
} // namespace popart

#endif // !GUARD_NEURALNET_INCREMENTMODX_HPP