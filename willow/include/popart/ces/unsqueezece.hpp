#ifndef GUARD_NEURALNET_CONSTEXPRS_UNSQUEEZECE_HPP
#define GUARD_NEURALNET_CONSTEXPRS_UNSQUEEZECE_HPP

#include <popart/ces/constexpr.hpp>

namespace popart {

class ConstExprUnsqueeze : public ConstExprOp {
public:
  ConstExprUnsqueeze(Op *);
  std::vector<char> compute() final;
};

} // namespace popart

#endif
