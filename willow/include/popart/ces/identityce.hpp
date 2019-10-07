#ifndef GUARD_NEURALNET_CONSTEXPRS_IDENTITYCE_HPP
#define GUARD_NEURALNET_CONSTEXPRS_IDENTITYCE_HPP

#include <popart/ces/constexpr.hpp>

namespace popart {

class ConstExprIdentity : public ConstExprOp {
public:
  ConstExprIdentity(Op *);
  std::vector<char> compute() final;
};

} // namespace popart

#endif
