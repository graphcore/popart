#ifndef GUARD_NEURALNET_CONSTEXPRS_SLICECE_HPP
#define GUARD_NEURALNET_CONSTEXPRS_SLICECE_HPP

#include <poponnx/ces/constexpr.hpp>
#include <poponnx/op/slice.hpp>

namespace poponnx {

class ConstExprSlice : public ConstExprOp {
public:
  ConstExprSlice(Op *);
  std::vector<char> compute() final;

private:
  std::vector<Slice> getAllSlices();
};

} // namespace poponnx

#endif
