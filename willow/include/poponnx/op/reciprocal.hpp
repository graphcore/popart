#ifndef GUARD_NEURALNET_RECIPROCAL_HPP
#define GUARD_NEURALNET_RECIPROCAL_HPP

#include <poponnx/op.hpp>
#include <poponnx/op/elementwise.hpp>

namespace poponnx {

class ReciprocalOp : public ElementWiseUnaryOp {
public:
  ReciprocalOp(const OperatorIdentifier &_opid,
               Ir *_ir,
               const std::string &name = "",
               const Attributes &_attr = {});
  std::unique_ptr<Op> clone() const final;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
};

class ReciprocalGradOp : public ElementWiseNonLinearUnaryGradOp {
public:
  ReciprocalGradOp(ReciprocalOp *);
  std::unique_ptr<Op> clone() const final;
};

} // namespace poponnx

#endif
