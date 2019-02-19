#ifndef GUARD_NEURALNET_CONSTEXPRS_TRANSPOSECE_HPP
#define GUARD_NEURALNET_CONSTEXPRS_TRANSPOSECE_HPP

#include <poponnx/ces/constexpr.hpp>

namespace poponnx {

class ConstExprTranspose : public ConstExprOp {
public:
  ConstExprTranspose(Op *op);
  std::vector<char> compute() final;
};

} // namespace poponnx

#endif
