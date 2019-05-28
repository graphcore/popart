#ifndef GUARD_NEURALNET_CEIL_HPP
#define GUARD_NEURALNET_CEIL_HPP

#include <poponnx/op/elementwise.hpp>

namespace poponnx {

class CeilOp : public ElementWiseUnaryOp {
public:
  CeilOp(const OperatorIdentifier &_opid, const Op::Settings &settings_);

  std::unique_ptr<Op> clone() const override;
  std::vector<std::unique_ptr<Op>> getGradOps() final;

  std::vector<std::tuple<OperatorIdentifier, float>>
  inplacePriorityDefault() const final;
  std::unique_ptr<Op> getInplaceVariant(const OperatorIdentifier &) const final;
};

class CeilInplaceOp : public ElementWiseInplaceUnaryOp {
public:
  CeilInplaceOp(const CeilOp &);
  std::unique_ptr<Op> clone() const final;
};

} // namespace poponnx

#endif
