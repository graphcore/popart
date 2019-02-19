#ifndef GUARD_NEURALNET_CONSTEXPRS_ADDCE_HPP
#define GUARD_NEURALNET_CONSTEXPRS_ADDCE_HPP

#include <poponnx/ces/constexpr.hpp>

namespace poponnx {

class ConstExprAdd : public ConstExprOp {
public:
  ConstExprAdd(Op *op);
  std::vector<char> compute() final;
};

} // namespace poponnx

#endif
