#ifndef GUARD_NEURALNET_EQUAL_HPP
#define GUARD_NEURALNET_EQUAL_HPP

#include <poponnx/op.hpp>
#include <poponnx/op/elementwise.hpp>

namespace poponnx {

class EqualOp : public BinaryComparisonOp {
public:
  EqualOp(const OperatorIdentifier &_opid, const Op::Settings &settings);
  std::unique_ptr<Op> clone() const final;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
};

} // namespace poponnx

#endif
