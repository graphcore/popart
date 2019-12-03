
#ifndef GUARD_NEURALNET_IDENTITY_HPP
#define GUARD_NEURALNET_IDENTITY_HPP

#include <popart/op.hpp>
#include <popart/op/elementwise.hpp>

namespace popart {

class IdentityOp : public ElementWiseUnaryOp {
public:
  IdentityOp(const OperatorIdentifier &_opid, const Op::Settings &settings);

  std::unique_ptr<Op> clone() const override;
  std::vector<std::unique_ptr<Op>> getGradOps() final;

  // For inplace support
  std::unique_ptr<Op>
  getInplaceVariant(const OperatorIdentifier &o) const final;
  std::vector<std::tuple<OperatorIdentifier, float>>
  inplacePriorityDefault() const final;
};

class IdentityInplaceOp : public IdentityOp {
public:
  IdentityInplaceOp(const OperatorIdentifier &_opid,
                    const Op::Settings &settings);
  IdentityInplaceOp(const IdentityOp &concatOp);

  std::unique_ptr<Op> clone() const override;

  view::Regions aliases(InIndex in, OutIndex) const final { return uses(in); }
};

class IdentityGradOp : public IdentityOp {
public:
  IdentityGradOp(const IdentityOp &fwdOp);
  std::unique_ptr<Op> clone() const final;

  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;

  static InIndex getInIndex() { return 0; }
  static OutIndex getOutIndex() { return 0; }
};

} // namespace popart

#endif
