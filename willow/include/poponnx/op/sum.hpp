#ifndef GUARD_NEURALNET_SUM_HPP
#define GUARD_NEURALNET_SUM_HPP

#include <poponnx/ir.hpp>

namespace willow {

class SumOp : public Op {
public:
  SumOp(const OpConstructorBundle &);
  void setup() final;
  std::unique_ptr<Op> clone() const final;
};
} // namespace willow

#endif
