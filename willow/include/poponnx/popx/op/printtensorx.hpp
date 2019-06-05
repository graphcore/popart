#ifndef GUARD_NEURALNET_PRINTTENSORX_HPP
#define GUARD_NEURALNET_PRINTTENSORX_HPP

#include <poponnx/names.hpp>
#include <poponnx/popx/opx.hpp>

namespace poponnx {

namespace popx {

class PrintTensorOpx : public Opx {
public:
  PrintTensorOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;

private:
  std::string getTitle() const;
};

} // namespace popx
} // namespace poponnx

#endif
