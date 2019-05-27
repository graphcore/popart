#ifndef GUARD_NEURALNET_CONSTEXPRS_GATHERCE_HPP
#define GUARD_NEURALNET_CONSTEXPRS_GATHERCE_HPP

#include <poponnx/ces/constexpr.hpp>

namespace poponnx {

class ConstExprGather : public ConstExprOp {
public:
  ConstExprGather(Op *op);
  std::vector<char> compute() final;
};

} // namespace poponnx

#endif
