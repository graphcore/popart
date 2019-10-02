#ifndef GUARD_NEURALNET_CONSTEXPRS_RESHAPECE_HPP
#define GUARD_NEURALNET_CONSTEXPRS_RESHAPECE_HPP

#include <popart/ces/identityce.hpp>

namespace popart {

class ConstExprReshape : public ConstExprIdentity {
public:
  ConstExprReshape(Op *op) : ConstExprIdentity(op) {}
};

} // namespace popart

#endif
