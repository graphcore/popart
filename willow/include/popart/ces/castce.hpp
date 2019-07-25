#ifndef GUARD_NEURALNET_CONSTEXPRS_CASTCE_HPP
#define GUARD_NEURALNET_CONSTEXPRS_CASTCE_HPP

#include <popart/ces/constexpr.hpp>

namespace popart {

class ConstExprCast : public ConstExprOp {
public:
  ConstExprCast(Op *op);
  std::vector<char> compute() final;
};
} // namespace popart

#endif
