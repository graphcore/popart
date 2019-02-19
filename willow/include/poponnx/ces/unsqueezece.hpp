#ifndef GUARD_NEURALNET_CONSTEXPRS_UNSQUEEZECE_HPP
#define GUARD_NEURALNET_CONSTEXPRS_UNSQUEEZECE_HPP

#include <poponnx/ces/constexpr.hpp>

namespace poponnx {

class ConstExprUnsqueeze : public ConstExprOp {
public:
  ConstExprUnsqueeze(Op *);
  std::vector<char> compute() final;
};

} // namespace poponnx

#endif
