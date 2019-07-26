#ifndef GUARD_NEURALNET_CONSTEXPRS_CONCATCE_HPP
#define GUARD_NEURALNET_CONSTEXPRS_CONCATCE_HPP

#include <popart/ces/constexpr.hpp>

namespace popart {

class ConstExprConcat : public ConstExprOp {
public:
  ConstExprConcat(Op *);
  std::vector<char> compute() final;

private:
  int64_t input_count;
  int64_t axis;
};

} // namespace popart

#endif
