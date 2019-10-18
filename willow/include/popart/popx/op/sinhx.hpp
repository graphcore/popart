#ifndef GUARD_NEURALNET_SINHX_HPP
#define GUARD_NEURALNET_SINHX_HPP

#include <popart/names.hpp>
#include <popart/popx/op/elementwisex.hpp>

namespace popart {
namespace popx {

class SinhComputex : public EwuComputex {

public:
  SinhComputex() = default;

  poplar::Tensor outplace(poplar::program::Sequence &,
                          poplar::Graph &,
                          const poplar::Tensor &,
                          const std::string &) const final;

  void inplace(poplar::program::Sequence &,
               poplar::Graph &,
               const poplar::Tensor &,
               const std::string &) const final;

  static std::unique_ptr<EwuComputex> get() {
    return std::unique_ptr<EwuComputex>(new SinhComputex);
  }
};

class SinhOpx : public ElementWiseUnaryOutplaceOpx {
public:
  SinhOpx(Op *, Devicex *);
};

class SinhInplaceOpx : public ElementWiseUnaryInplaceOpx {
public:
  SinhInplaceOpx(Op *, Devicex *);
};

class SinhGradOpx : public Opx {
public:
  SinhGradOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif
