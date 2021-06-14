// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_DYNAMICBASE_HPP
#define GUARD_NEURALNET_DYNAMICBASE_HPP

#include <popart/op.hpp>

// Class hierarchy
// DynamicBaseOp
//   DynamicSliceBaseOp
//     DynamicSliceOp
//     DynamicUpdateUpdateGradOp
//   DynamicSlicePadGradOp
//   DynamicBinaryBaseOp
//     DynamicTernaryBaseOp
//       DynamicAddOp
//       DynamicUpdateOp
//       DynamicTernaryBaseInplaceOp
//         DynamicAddInplaceOp
//         DynamicUpdateInplaceOp
//     DynamicBinaryBaseInplaceOp
//       DynamicZeroInplaceOp
//     DynamicZeroOp
//     DynamicUpdateToUpdateGradOp
//     DynamicZeroGradOp

namespace popart {

// Base Ops
class DynamicBaseOp : public Op {
public:
  DynamicBaseOp(const OperatorIdentifier &_opid,
                std::vector<int64_t> axes_,
                std::vector<int64_t> sizes_,
                bool noOverlap_,
                const Op::Settings &);
  void setup() override;

  static InIndex getIndexInIndex() { return 1; }
  static OutIndex getOutIndex() { return 0; }

  float getSubgraphValue() const final { return getHighSubgraphValue(); }

  const std::vector<int64_t> &getAxes() const { return axes; }
  void setAxes(const std::vector<int64_t> &x) { axes = x; }

  const std::vector<int64_t> &getSizes() const { return sizes; }
  void setSizes(const std::vector<int64_t> &x) { sizes = x; }

  bool isNotOverlapping() const { return noOverlap; }

  TensorInfo createOutInfo() const;
  void appendOutlineAttributes(OpSerialiserBase &) const override;

protected:
  std::vector<int64_t> axes;
  std::vector<int64_t> sizes;

  // If set to true, then correct gradient backpropagation is only guaranteed if
  // each region in the output tensor has only exactly one populator
  // (operation that writes data to this region).
  // There are no run-time or compile-time checks possible to ensure this.
  bool noOverlap;
};

class DynamicSliceBaseOp : public DynamicBaseOp {
public:
  DynamicSliceBaseOp(const OperatorIdentifier &_opid,
                     std::vector<int64_t> axes_,
                     std::vector<int64_t> sizes_,
                     bool noOverlap_,
                     const Op::Settings &);
  void setup() final;

  TensorInfo createOutInfo() const;
  static InIndex getInIndex() { return 0; }
};

class DynamicBinaryBaseOp : public DynamicBaseOp {
public:
  DynamicBinaryBaseOp(const OperatorIdentifier &_opid,
                      std::vector<int64_t> axes_,
                      std::vector<int64_t> sizes_,
                      bool noOverlap_,
                      const Op::Settings &settings_,
                      TensorInfo updateInInfo_ = TensorInfo());
  void setup() final;

  const TensorInfo &getUpdateTensorInfo() const { return updateInInfo; }

  static InIndex getUpdateInIndex() { return 0; }
  static InIndex getIndexInIndex() { return 1; }
  static OutIndex getOutIndex() { return 0; }

  virtual void growAliasModel(AliasModel &m) const final;

  poprithms::memory::inplace::Proposal
  mapInplaceProposal(const AliasModel &, OperatorIdentifier) const override;

protected:
  TensorInfo updateInInfo;
};

class DynamicBinaryBaseInplaceOp : public DynamicBinaryBaseOp {
public:
  DynamicBinaryBaseInplaceOp(const OperatorIdentifier &_opid,
                             std::vector<int64_t> axes_,
                             std::vector<int64_t> sizes_,
                             bool noOverlap_,
                             const Op::Settings &settings_,
                             TensorInfo updateInInfo_ = TensorInfo());

  view::RegMap fwdRegMap(InIndex, OutIndex) const final;
  view::RegMap bwdRegMap(InIndex, OutIndex) const final;

  // This Op aliases and modifies the input
  view::Regions aliases(InIndex, OutIndex) const final;
  view::Regions modifies(InIndex) const final;
};

class DynamicTernaryBaseOp : public DynamicBinaryBaseOp {
public:
  DynamicTernaryBaseOp(const OperatorIdentifier &_opid,
                       std::vector<int64_t> axes_,
                       std::vector<int64_t> sizes_,
                       bool noOverlap_,
                       const Op::Settings &settings_,
                       TensorInfo updateInInfo_ = TensorInfo());
  static InIndex getUpdateInIndex() { return 0; }
  static InIndex getInIndex() { return 2; }
};

class DynamicTernaryBaseInplaceOp : public DynamicTernaryBaseOp {
public:
  DynamicTernaryBaseInplaceOp(const OperatorIdentifier &_opid,
                              std::vector<int64_t> axes_,
                              std::vector<int64_t> sizes_,
                              bool noOverlap_,
                              const Op::Settings &settings_,
                              TensorInfo updateInInfo_ = TensorInfo());

  view::RegMap fwdRegMap(InIndex, OutIndex) const final;
  view::RegMap bwdRegMap(InIndex, OutIndex) const final;

  // This Op aliases and modifies the input
  view::Regions aliases(InIndex, OutIndex) const final;
  view::Regions modifies(InIndex) const final;
};

} // namespace popart

#endif
