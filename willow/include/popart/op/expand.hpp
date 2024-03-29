// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OP_EXPAND_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_EXPAND_HPP_

#include <cstddef>
#include <map>
#include <memory>
#include <tuple>
#include <vector>
#include <poprithms/memory/inplace/proposal.hpp>
#include <popart/op.hpp>
#include <popart/tensorindex.hpp>

#include "popart/graphcoreoperators.hpp"
#include "popart/names.hpp"
#include "popart/operatoridentifier.hpp"
#include "popart/region.hpp" // IWYU pragma: keep
#include "popart/tensordebuginfo.hpp"
#include "popart/tensorinfo.hpp"

namespace popart {
class AliasModel;

class ExpandOp : public Op {
public:
  ExpandOp(const OperatorIdentifier &_opid, const Op::Settings &settings_)
      : Op(_opid, settings_) {}
  ExpandOp(const OperatorIdentifier &_opid,
           const Shape &_outShape,
           const Op::Settings &settings);

  std::unique_ptr<Op> clone() const override;
  void setup() final;

  std::vector<std::unique_ptr<Op>> getGradOps() final;

  Shape getOutShape() const { return outShape; }

  // note that this is not final, ExpandInplaceOp overrides it
  std::vector<std::tuple<OperatorIdentifier, float>>
  inplacePriorityDefault() const override {
    return {{Onnx::CustomOperators::ExpandInplace, 10.0f}};
  }

  std::unique_ptr<Op>
  getInplaceVariant(const OperatorIdentifier &) const override;

  static InIndex getInTensorIndex() { return 0; }
  static InIndex getInShapeIndex() { return 1; }
  static OutIndex getOutIndex() { return 0; }

  view::RegMap fwdRegMap(InIndex, OutIndex) const final;
  view::RegMap bwdRegMap(InIndex, OutIndex) const final;

  bool canBeReplacedByIdentity() const override {
    return input->getIndexShapeMap()[ExpandOp::getInTensorIndex()] == outShape;
  }

  void growAliasModel(AliasModel &) const override;

  poprithms::memory::inplace::Proposal
  mapInplaceProposal(const AliasModel &, OperatorIdentifier) const override;

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

  void connectInTensor(InIndex inIndex, TensorId tenId) final;

private:
  Shape outShape;

  void regMapPreChecks(InIndex inIndex) const;
  void finaliseShape();
};

// An inplace variant of the expand op
class ExpandInplaceOp : public ExpandOp {
public:
  ExpandInplaceOp(const OperatorIdentifier &_opid,
                  const Shape &,
                  const Op::Settings &settings_);
  ExpandInplaceOp(const ExpandOp &expandOp);

  std::unique_ptr<Op> clone() const override;

  std::vector<std::tuple<OperatorIdentifier, float>>
  inplacePriorityDefault() const final {
    return {};
  }

  std::unique_ptr<Op>
  getInplaceVariant(const OperatorIdentifier &o) const final {
    // this throws an error
    return Op::getInplaceVariant(o);
  }

  // The whole of the used area is aliased. "modifies" is still empty
  view::Regions aliases(InIndex in, OutIndex) const final { return uses(in); }
};

class ExpandGradOp : public Op {
public:
  ExpandGradOp(const ExpandOp &op);
  ExpandGradOp(const ExpandInplaceOp &op);

  std::unique_ptr<Op> clone() const override;
  void setup() override;

  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;

  static InIndex getDYIndex() { return 0; }
  static OutIndex getOutIndex() { return 0; }

  std::vector<size_t> getXShape() { return gradInfo.shape_szt(); }
  float getSubgraphValue() const final { return getLowSubgraphValue(); }

protected:
  // An unsafe constructor that allows using any OperatorIdentifier
  ExpandGradOp(const OperatorIdentifier &_opid,
               const ExpandGradOp &expand_grad_op);

private:
  InIndex fwdInput;

  TensorInfo gradInfo;
  std::map<int, int> gradOutToNonGradInInfo;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_OP_EXPAND_HPP_
