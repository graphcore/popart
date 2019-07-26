#ifndef GUARD_NEURALNET_AND_HPP
#define GUARD_NEURALNET_AND_HPP

#include <popart/op.hpp>
#include <popart/op/elementwise.hpp>

namespace popart {

class AndOp : public BinaryComparisonOp {
public:
  AndOp(const OperatorIdentifier &_opid, const Op::Settings &settings);
  std::unique_ptr<Op> clone() const final;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
};

} // namespace popart

#endif
