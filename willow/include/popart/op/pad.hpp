// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_PAD_HPP
#define GUARD_NEURALNET_PAD_HPP

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <vector>
#include <poprithms/memory/inplace/proposal.hpp>
#include <popart/op.hpp>

#include "popart/names.hpp"
#include "popart/region.hpp"

namespace popart {
class AliasModel;
class OpSerialiserBase;
struct OperatorIdentifier;
struct Slice;

class BasePadOp : public Op {
public:
  BasePadOp(const OperatorIdentifier &_opid,
            const std::vector<int64_t> &_pads,
            const std::vector<unsigned> &_flips,
            float value_,
            const std::string &_mode,
            const Op::Settings &settings_);
  std::unique_ptr<Op> clone() const override;

  // returns true if all pad sizes in all dimensions
  // and on all sides, are zero. i.e. no padding.
  bool padSizeZero() const;

  static InIndex getInIndex() { return 0; }
  static OutIndex getOutIndex() { return 0; }

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

  // The region of the output tensors which is based on the input tensor value.
  // The complement of this region is the padding region.
  view::Region valueRegion() const;

  // The dimensions along which there is non-zero padding, in ascending order.
  std::vector<int64_t> padDimensions() const;

  // The amount of padding to prepend at the start of the input Tensor.
  int64_t getLowerPadding(size_t dim) const { return pads[dim]; }

  // The amount of padding to append at the end of the input Tensor.
  int64_t getUpperPadding(size_t dim) const { return pads[dim + getRank()]; }

  // The padding mode, one of "constant", "edge", and "reflect".
  const std::string &getMode() const { return mode; }

  // For constant mode padding, the value to pad with.
  float getPadValue() const { return pad_value; }

  void appendOutlineAttributes(OpSerialiserBase &) const override;

  void setup() final;

  view::RegMap fwdRegMap(InIndex, OutIndex) const final;
  view::RegMap bwdRegMap(InIndex, OutIndex) const final;

  // The rank of the input (and output) Tensor.
  int64_t getRank() const { return pads.size() / 2; }

  std::vector<Slice> getSlices() const;

  // All of the lower padding sizes
  std::vector<std::ptrdiff_t> getLowerPadding() const { return getPadRange(0); }

  // All of the upper padding sizes.
  std::vector<std::ptrdiff_t> getUpperPadding() const {
    return getPadRange(getRank());
  }

  const std::vector<int64_t> &getPads() const { return pads; }
  const std::vector<unsigned> &getFlips() const { return flips; }

  virtual void growAliasModel(AliasModel &) const override;

private:
  // all lower and upper padding values. The first inRank values are the lower
  // padding values, the final inRank values are the upper padding values.
  std::vector<int64_t> pads;

  // A vector of axes along which to flip the input tensor before padding
  std::vector<unsigned> flips;

  float pad_value;
  std::string mode;

  std::vector<std::ptrdiff_t> getPadRange(size_t startIndex) const;
  void runtimeConfirmShapes() const;
};

class BasePadOutplaceOp : public BasePadOp {
public:
  BasePadOutplaceOp(const OperatorIdentifier &_opid,
                    const std::vector<int64_t> &_pads,
                    const std::vector<unsigned> &_flips,
                    float value_,
                    const std::string &_mode,
                    const Op::Settings &settings_);
  std::unique_ptr<Op> clone() const override;

  bool canBeReplacedByIdentity() const override { return padSizeZero(); }

  poprithms::memory::inplace::Proposal
  mapInplaceProposal(const AliasModel &, OperatorIdentifier) const override;

  std::vector<std::tuple<OperatorIdentifier, float>>
  inplacePriorityDefault() const override;

  std::unique_ptr<Op>
  getInplaceVariant(const OperatorIdentifier &) const override;
};

class PadOp : public BasePadOutplaceOp {
public:
  PadOp(const OperatorIdentifier &_opid,
        const std::vector<int64_t> &_pads,
        const std::vector<unsigned> &_flips,
        float value_,
        const std::string &_mode,
        const Op::Settings &settings_);

  std::unique_ptr<Op> clone() const final;
  std::vector<std::unique_ptr<Op>> getGradOps() final;

  void connectInTensor(InIndex inIndex, TensorId tenId) final;
};

class PadInplaceOp : public BasePadOp {
public:
  PadInplaceOp(const BasePadOutplaceOp &);
  std::unique_ptr<Op> clone() const final;

  std::unique_ptr<Op>
  getInplaceVariant(const OperatorIdentifier &o) const final {
    // this throws an error
    return Op::getInplaceVariant(o);
  }

  // PadInplace does not modify any input values, so the "modifies" function is
  // left unchanged.
  view::Regions aliases(InIndex, OutIndex) const override;
  view::Regions uses(InIndex index) const override;
};

} // namespace popart

#endif
