#ifndef GUARD_NEURALNET_RECIPROCAL_HPP
#define GUARD_NEURALNET_RECIPROCAL_HPP

#include <poponnx/op.hpp>
#include <poponnx/op/elementwise.hpp>

namespace poponnx {

class ReciprocalOp : public ElementWiseUnaryOp {
public:
  ReciprocalOp(const OperatorIdentifier &_opid, const Op::Settings &settings_);
  std::unique_ptr<Op> clone() const final;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
};

class ReciprocalGradOp : public ElementWiseNonLinearUnaryGradOp {
public:
  ReciprocalGradOp(const ReciprocalOp &);
  std::unique_ptr<Op> clone() const final;
};

} // namespace poponnx

#endif
