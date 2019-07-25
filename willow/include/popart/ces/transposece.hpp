#ifndef GUARD_NEURALNET_CONSTEXPRS_TRANSPOSECE_HPP
#define GUARD_NEURALNET_CONSTEXPRS_TRANSPOSECE_HPP

#include <popart/ces/constexpr.hpp>

namespace popart {

class ConstExprTranspose : public ConstExprOp {
public:
  ConstExprTranspose(Op *op);
  std::vector<char> compute() final;
};

} // namespace popart

#endif
