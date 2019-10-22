#ifndef GUARD_NEURALNET_CONSTEXPRS_UNSQUEEZECE_HPP
#define GUARD_NEURALNET_CONSTEXPRS_UNSQUEEZECE_HPP

#include <popart/ces/identityce.hpp>

namespace popart {

class ConstExprUnsqueeze : public ConstExprIdentity {
public:
  ConstExprUnsqueeze(Op *op_) : ConstExprIdentity(op_) {}
};

} // namespace popart

#endif
