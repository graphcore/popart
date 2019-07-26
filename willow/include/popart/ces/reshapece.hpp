#ifndef GUARD_NEURALNET_CONSTEXPRS_RESHAPECE_HPP
#define GUARD_NEURALNET_CONSTEXPRS_RESHAPECE_HPP

#include <popart/ces/constexpr.hpp>

namespace popart {

class ConstExprReshape : public ConstExprOp {
public:
  ConstExprReshape(Op *op);
  std::vector<char> compute() final;
};

} // namespace popart

#endif
