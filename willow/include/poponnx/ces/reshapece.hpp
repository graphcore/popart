#ifndef GUARD_NEURALNET_CONSTEXPRS_RESHAPECE_HPP
#define GUARD_NEURALNET_CONSTEXPRS_RESHAPECE_HPP

#include <poponnx/ces/constexpr.hpp>

namespace poponnx {

class ConstExprReshape : public ConstExprOp {
public:
  ConstExprReshape(Op *op);
  std::vector<char> compute() final;
};

} // namespace poponnx

#endif
