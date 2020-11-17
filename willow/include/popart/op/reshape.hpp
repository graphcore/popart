// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_RESHAPE_HPP
#define GUARD_NEURALNET_RESHAPE_HPP

#include <popart/op.hpp>

namespace popart {

// TODO: merge Reshape and Squeeze functionality (T5886)

// This Op is based on the ONNX Operator described at
// github.com/onnx/onnx/blob/master/docs/Operators.md#Reshape
// but it is slightly different: this Op is static w.r.t. shape

class ReshapeBaseOp : public Op {
public:
  ReshapeBaseOp(const OperatorIdentifier &_opid,
                const Shape &,
                const Op::Settings &settings_);

  ReshapeBaseOp(const OperatorIdentifier &_opid, const Op::Settings &settings_)
      : Op(_opid, settings_) {}

  void setup() final;

  static InIndex getInIndex() { return 0; }
  static OutIndex getOutIndex() { return 0; }

  void setOutShape(const Shape &value);
  const Shape &getOutShape() const;

  view::RegMap fwdRegMap(InIndex, OutIndex) const final;
  view::RegMap bwdRegMap(InIndex, OutIndex) const final;

  bool canBeReplacedByIdentity() const override;

  float getSubgraphValue() const final { return getLowSubgraphValue(); }
  void connectInTensor(InIndex inIndex, TensorId tenId) final;

  bool canShard() const override { return true; }
  void configureShardedOp(Op *const shardedOp,
                          const Settings *const settings_) const override;

protected:
  // The shape of the data output tensor
  Shape outShape;

  void finaliseShape();
};

class ReshapeOp : public ReshapeBaseOp {
public:
  ReshapeOp(const OperatorIdentifier &_opid,
            const Shape &s,
            const Op::Settings &settings_)
      : ReshapeBaseOp(_opid, s, settings_) {}

  ReshapeOp(const OperatorIdentifier &_opid, const Op::Settings &settings_)
      : ReshapeBaseOp(_opid, settings_) {}

  std::unique_ptr<Op> clone() const override;

  std::vector<std::unique_ptr<Op>> getGradOps() final;

  std::unique_ptr<Op>
  getInplaceVariant(const OperatorIdentifier &o) const final;

  std::vector<std::tuple<OperatorIdentifier, float>>
  inplacePriorityDefault() const final {
    return {{Onnx::CustomOperators::ReshapeInplace, 10}};
  }
};

class ReshapeInplaceOp : public ReshapeBaseOp {
public:
  ReshapeInplaceOp(const OperatorIdentifier &_opid,
                   const Shape &,
                   const Op::Settings &settings_);
  ReshapeInplaceOp(const ReshapeOp &);
  std::unique_ptr<Op> clone() const final;

  std::unique_ptr<Op>
  getInplaceVariant(const OperatorIdentifier &o) const final {
    // this throws an error
    return Op::getInplaceVariant(o);
  }

  // modifies and uses are still the defaults, but aliases changes
  // to be the same as uses (the full out region)
  view::Regions aliases(InIndex in, OutIndex) const final { return uses(in); }
};

// The gradient of reshape is the reverse of the
// reshape (which is a reshape)
class ReshapeGradOp : public ReshapeOp {
public:
  ReshapeGradOp(const ReshapeOp &);
  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;
};

} // namespace popart

#endif
