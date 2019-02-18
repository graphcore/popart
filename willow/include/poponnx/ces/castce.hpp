#ifndef GUARD_NEURALNET_CONSTEXPRS_CASTCE_HPP
#define GUARD_NEURALNET_CONSTEXPRS_CASTCE_HPP

#include <poponnx/ces/constexpr.hpp>

namespace poponnx {

class ConstExprCast : public ConstExprOp {
public:
  ConstExprCast(Op *op);
  std::vector<char> compute() final;
};
} // namespace poponnx

#endif
