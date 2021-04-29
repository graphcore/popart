#ifndef GUARD_NEURALNET_DETACH_HPP
#define GUARD_NEURALNET_DETACH_HPP

#include <popart/op.hpp>
#include <popart/op/elementwise.hpp>
#include <popart/opmanager.hpp>

namespace popart {

class DetachOp : public ElementWiseUnaryOp {
public:
  DetachOp(const OperatorIdentifier &_opid, const Op::Settings &settings);

  std::vector<std::unique_ptr<Op>> getGradOps() final { return {}; }
  std::unique_ptr<Op> clone() const override;

  // For inplace support
  std::unique_ptr<Op>
  getInplaceVariant(const OperatorIdentifier &o) const final;
  std::vector<std::tuple<OperatorIdentifier, float>>
  inplacePriorityDefault() const final;

  // mathematically equivalent to identity, but we don't want it to be replaced
  // by identity because it's behaviour is different (no backwards pass Op for
  // Detach)
  bool isIdentity() const final { return true; }
};

class DetachInplaceOp : public DetachOp {
public:
  DetachInplaceOp(const DetachOp &detachOp);

  std::unique_ptr<Op> clone() const override;

  view::Regions aliases(InIndex in, OutIndex) const final { return uses(in); }
};

} // namespace popart

#endif
