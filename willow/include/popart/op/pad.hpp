#ifndef GUARD_NEURALNET_PAD_HPP
#define GUARD_NEURALNET_PAD_HPP

#include <popart/op.hpp>

namespace popart {

class BasePadOp : public Op {
public:
  BasePadOp(const OperatorIdentifier &_opid,
            const std::vector<int64_t> &_pads,
            float value_,
            const std::string &_mode,
            const Op::Settings &settings_);

  // returns true if all pad sizes in all dimensions
  // and on all sides, are zero
  bool padSizeZero() const;

  static InIndex getInIndex() { return 0; }
  static OutIndex getOutIndex() { return 0; }

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

  // The region of the output tensors which is based on the input tensor value.
  // The complement of this region is the padding region.
  view::Region valueRegion() const;
  std::vector<int64_t> padDimensions() const;

  const std::vector<int64_t> &getPads() const;
  float getPadValue() const;
  const std::string &getMode() const;

  void appendOutlineAttributes(OpSerialiserBase &) const override;

  void setup() final;

  view::RegMap fwdRegMap(InIndex, OutIndex) const final;
  view::RegMap bwdRegMap(InIndex, OutIndex) const final;

private:
  std::vector<int64_t> pads;
  float pad_value;
  std::string mode;
};

class PadOp : public BasePadOp {
public:
  PadOp(const OperatorIdentifier &_opid,
        const std::vector<int64_t> &_pads,
        float value_,
        const std::string &_mode,
        const Op::Settings &settings_);

  std::unique_ptr<Op> clone() const final;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
  bool canBeReplacedByIdentity() override;

  std::vector<std::tuple<OperatorIdentifier, float>>
  inplacePriorityDefault() const override;

  std::unique_ptr<Op>
  getInplaceVariant(const OperatorIdentifier &) const override;
};

class PadInplaceOp : public BasePadOp {
public:
  PadInplaceOp(const PadOp &);
  std::unique_ptr<Op> clone() const final;

  std::unique_ptr<Op>
  getInplaceVariant(const OperatorIdentifier &o) const final {
    // this throws an error
    return Op::getInplaceVariant(o);
  }

  view::Regions modifies(InIndex) const override;
  view::Regions aliases(InIndex, OutIndex) const override;
  view::Regions uses(InIndex index) const override;
};

} // namespace popart

#endif
