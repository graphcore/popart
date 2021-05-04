// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_ELEMENTWISEUNARY_HPP
#define GUARD_NEURALNET_ELEMENTWISEUNARY_HPP

#include <map>
#include <memory>
#include <popart/op.hpp>

namespace popart {
class ElementWiseBinaryBaseOp;
view::RegMap binaryFwdRegMapImpl(const ElementWiseBinaryBaseOp &op,
                                 InIndex argIndex);

view::RegMap binaryBwdRegMapImpl(const ElementWiseBinaryBaseOp &op,
                                 InIndex argIndex);

// Base class for elementwise unary operations
class ElementWiseUnaryOp : public Op {
public:
  ElementWiseUnaryOp(const OperatorIdentifier &_opid,
                     const Op::Settings &_settings);
  void setup() final;

  static InIndex getInIndex() { return 0; }
  static OutIndex getOutIndex() { return 0; }

  // Making this function override and not final, as there
  // may be a more / less expensive to compute non-linearity.
  float getSubgraphValue() const override { return getLowSubgraphValue(); }

  bool canShard() const override { return true; }

  void setProposal(poprithms::memory::inplace::Proposal &,
                   const PoprithmsAliaser &,
                   OperatorIdentifier) const override;

  ReplicatedTensorShardingIndices
  getReplicatedTensorShardingIndices() const override;

  virtual void growAliaser(PoprithmsAliaser &) const override;

  /**
   * \return true, if and only if (iff) this Op is mathematically equivalent to
   *        f(x) = x. This is slightly different to canBeReplacedByIdentity; for
   *        example Detach and Identity have isIdentity overriden to return
   *        true, but still return false for canBeReplacedByIdentity.
   */
  virtual bool isIdentity() const { return canBeReplacedByIdentity(); }
};

// Base class for elementwise unary boolean output operations
class ElementWiseUnaryBooleanOp : public Op {
public:
  ElementWiseUnaryBooleanOp(const OperatorIdentifier &_opid,
                            const Op::Settings &_settings);
  void setup() final;

  static InIndex getInIndex() { return 0; }
  static OutIndex getOutIndex() { return 0; }

  // Making this function override and not final, as there
  // may be a more / less expensive to compute non-linearity.
  float getSubgraphValue() const override { return getLowSubgraphValue(); }

  bool canShard() const override { return true; }
};

class ElementWiseInplaceUnaryOp : public ElementWiseUnaryOp {
public:
  ElementWiseInplaceUnaryOp(const OperatorIdentifier &_opid,
                            const Op::Settings &_settings)
      : ElementWiseUnaryOp(_opid, _settings) {}

  view::Regions modifies(InIndex index) const final { return uses(index); }
  view::Regions aliases(InIndex in, OutIndex) const final { return uses(in); }
  // "uses" is still the full region
  // "fwdRegMap" is still the identity
  // "bwdRegMap" is still the identity
};

// Base class for gradients of element-wise, non-linear, unary operations
// Non-linear elementwise ops gradients take both the input, and gradient
// output of the corresponding forward operation as inputs.
class ElementWiseNonLinearUnaryGradOp : public Op {
public:
  ElementWiseNonLinearUnaryGradOp(const OperatorIdentifier &_opid,
                                  const ElementWiseUnaryOp &fwdOp);
  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;
  void setup() final;

  // This grad op takes 2 inputs,
  // 1) the gradient of the output of the corresponding forward op, and
  // 2) the input of the forward op.
  // The indices at which these two tensors are inputs to this grad op are
  // getGradInIndex and getFwdArgInIndex respectively.
  static InIndex getGradInIndex() { return 0; }
  static InIndex getFwdArgInIndex() { return 1; }
  static OutIndex getOutIndex() { return 0; }

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

  bool canShard() const override { return true; }
};

// Common base class for elementwise binary operations
class ElementWiseBinaryBaseOp : public Op {
public:
  ElementWiseBinaryBaseOp(const OperatorIdentifier &_opid,
                          const Op::Settings &_settings);
  void setup() final;

  // Current implementation places arg0 input at index 0, and arg1 input
  // at index 1.
  static InIndex getArg0InIndex() { return 0; }
  static InIndex getArg1InIndex() { return 1; }
  static OutIndex getOutIndex() { return 0; }

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

  bool canShard() const override { return true; }

  virtual void growAliaser(PoprithmsAliaser &) const override;

  ReplicatedTensorShardingIndices
  getReplicatedTensorShardingIndices() const override;

  view::RegMap fwdRegMap(InIndex argIndex, OutIndex) const final {
    return binaryFwdRegMapImpl(*this, argIndex);
  }

  view::RegMap bwdRegMap(InIndex argIndex, OutIndex) const final {
    return binaryBwdRegMapImpl(*this, argIndex);
  }
};

// Base class for non-inplace elementwise binary operation, which may have
// registered inplace variants for either LHS or RHS.
class ElementWiseBinaryOp : public ElementWiseBinaryBaseOp {
public:
  ElementWiseBinaryOp(const OperatorIdentifier &_opid,
                      const Op::Settings &_settings);

  std::vector<std::tuple<OperatorIdentifier, float>>
  inplacePriorityDefault() const final;

  std::unique_ptr<Op> getInplaceVariant(const OperatorIdentifier &) const final;

  void setInplacePriority(const OperatorIdentifier &, float);
  float getInplacePriority(const OperatorIdentifier &) const;

  void setProposal(poprithms::memory::inplace::Proposal &,
                   const PoprithmsAliaser &,
                   OperatorIdentifier) const override;

private:
  virtual bool hasLhsInplaceVariant() const;
  virtual bool hasRhsInplaceVariant() const;

  virtual std::unique_ptr<Op> getLhsInplaceVariant() const;
  virtual std::unique_ptr<Op> getRhsInplaceVariant() const;

  virtual OperatorIdentifier getLhsOperatorIdentifier() const;
  virtual OperatorIdentifier getRhsOperatorIdentifier() const;

  std::map<OperatorIdentifier, float> inplacePriorities;
};

// Base class for a non-inplace elementwise binary operation which is numpy
// broadcastable and has grad ops for the two arguments
template <class Arg0GradOp, class Arg1GradOp>
class ElementWiseNpBroadcastableBinaryWithGradOp : public ElementWiseBinaryOp {
public:
  ElementWiseNpBroadcastableBinaryWithGradOp(const OperatorIdentifier &_opid,
                                             const Op::Settings &_settings)
      : ElementWiseBinaryOp(_opid, _settings) {}

  virtual std::vector<std::unique_ptr<Op>> getGradOps() final {
    std::vector<std::unique_ptr<Op>> upops;

    const auto &shape_in_0   = inShape(getArg0InIndex());
    const auto &shape_in_1   = inShape(getArg1InIndex());
    const auto &shape_output = outShape(getOutIndex());

    upops.emplace_back(
        new Arg0GradOp(*this, npReductionAxis(shape_in_0, shape_output)));
    upops.emplace_back(
        new Arg1GradOp(*this, npReductionAxis(shape_in_1, shape_output)));
    return upops;
  }
};

// Base class for inplace LHS for elementwise binary operations
template <class Derived>
class ElementWiseBinaryInplaceLhsOp : public ElementWiseBinaryBaseOp {
public:
  ElementWiseBinaryInplaceLhsOp(const OperatorIdentifier &_opid,
                                const Op::Settings &_settings)
      : ElementWiseBinaryBaseOp(_opid, _settings) {}

  view::Regions modifies(InIndex index) const final {
    if (index == getArg0InIndex()) {
      return {view::Region::getFull(inShape(index))};
    } else if (index == getArg1InIndex()) {
      return {view::Region::getEmpty(inRank(index))};
    } else {
      throw error("Invalid index passed to modifies method for Operator {}",
                  opid);
    }
  }

  view::Regions aliases(InIndex index, OutIndex) const final {
    if (index == getArg0InIndex()) {
      return {view::Region::getFull(inShape(index))};
    } else if (index == getArg1InIndex()) {
      return {view::Region::getEmpty(inRank(index))};
    } else {
      throw error("Invalid index passed to aliases method for Operator {}",
                  opid);
    }
  }

  std::unique_ptr<Op> clone() const final {
    return std::unique_ptr<Derived>(new Derived(getSettings()));
  }
};

// Base class for inplace RHS for elementwise binary operations
template <class Derived>
class ElementWiseBinaryInplaceRhsOp : public ElementWiseBinaryBaseOp {
public:
  ElementWiseBinaryInplaceRhsOp(const OperatorIdentifier &_opid,
                                const Op::Settings &_settings)
      : ElementWiseBinaryBaseOp(_opid, _settings) {}

  view::Regions modifies(InIndex index) const final {
    if (index == getArg0InIndex()) {
      return {view::Region::getEmpty(inRank(index))};
    } else if (index == getArg1InIndex()) {
      return {view::Region::getFull(inShape(index))};
    } else {
      throw error("Invalid index passed to modifies method for Operator {}",
                  opid);
    }
  }

  view::Regions aliases(InIndex index, OutIndex) const final {
    if (index == getArg0InIndex()) {
      return {view::Region::getEmpty(inRank(index))};
    } else if (index == getArg1InIndex()) {
      return {view::Region::getFull(inShape(index))};
    } else {
      throw error("Invalid index passed to aliases method for Operator {}",
                  opid);
    }
  }

  std::unique_ptr<Op> clone() const final {
    return std::unique_ptr<Derived>(new Derived(getSettings()));
  }
};

// Base class for gradients of element-wise, binary operations
// Numpy-style broadcasting is assumed
class ElementWiseBinaryGradOp : public Op {
public:
  ElementWiseBinaryGradOp(const OperatorIdentifier &_opid,
                          const std::vector<int64_t> &_reduction_axes,
                          const TensorInfo &_forward_op_arg_info,
                          const Op::Settings &_settings);
  void setup() final;

  // Returns the axes along which to perform the reduction, as set in the
  // constructor, for caess of numpy-style broadcasting
  const std::vector<int64_t> &getReductionAxes() const {
    return reduction_axes;
  }

  // Low subgraph value is assumed by default
  float getSubgraphValue() const final { return getLowSubgraphValue(); }

  static InIndex getGradInIndex() { return 0; }
  static InIndex getFwdArg0InIndex() { return 1; }
  static InIndex getFwdArg1InIndex() { return 2; }
  static InIndex getFwdOutIndex() { return 3; }
  static OutIndex getOutIndex() { return 0; }

  const std::map<int, int> &gradOutToNonGradIn() const final {
    if (isArg1Grad()) {
      static const std::map<int, int> outInfo = {
          {getOutIndex(), ElementWiseBinaryBaseOp::getArg1InIndex()}};
      return outInfo;
    } else {
      static const std::map<int, int> outInfo = {
          {getOutIndex(), ElementWiseBinaryBaseOp::getArg0InIndex()}};
      return outInfo;
    }
  }

  // Set up both inputs outputs and grad for output
  // (Not all are used for all ops, but this is not an issue as they are
  // replaced by a pattern)
  virtual const std::vector<GradInOutMapper> &gradInputInfo() const final {
    static const std::vector<GradInOutMapper> inInfo = {
        {getGradInIndex(),
         ElementWiseBinaryBaseOp::getOutIndex(),
         GradOpInType::GradOut},
        {getFwdArg0InIndex(),
         ElementWiseBinaryBaseOp::getArg0InIndex(),
         GradOpInType::In},
        {getFwdArg1InIndex(),
         ElementWiseBinaryBaseOp::getArg1InIndex(),
         GradOpInType::In},
        {getFwdOutIndex(),
         ElementWiseBinaryBaseOp::getOutIndex(),
         GradOpInType::Out}};

    return inInfo;
  }

protected:
  // Returns true if Arg1 grad, false otherwise
  virtual bool isArg1Grad() const = 0;

private:
  // Used to set the outputs TensorInfo
  TensorInfo forward_op_arg_info;
  // reduction axes eventually passed to ReduceSumOp
  std::vector<int64_t> reduction_axes;
};

// Base class of ElementWiseBinaryGradOp for Arg0
template <class Derived>
class ElementWiseBinaryArg0GradOp : public ElementWiseBinaryGradOp {
public:
  ElementWiseBinaryArg0GradOp(const OperatorIdentifier &_opid,
                              const std::vector<int64_t> &_reduction_axes,
                              const TensorInfo &_forward_op_arg_info,
                              const Op::Settings &_settings)
      : ElementWiseBinaryGradOp(_opid,
                                _reduction_axes,
                                _forward_op_arg_info,
                                _settings) {}

  std::unique_ptr<Op> clone() const final {
    return std::unique_ptr<Derived>(
        new Derived(*dynamic_cast<const Derived *>(this)));
  }

protected:
  virtual bool isArg1Grad() const final { return false; }
};

// Base class of ElementWiseBinaryGradOp for Arg1
template <class Derived>
class ElementWiseBinaryArg1GradOp : public ElementWiseBinaryGradOp {
public:
  ElementWiseBinaryArg1GradOp(const OperatorIdentifier &_opid,
                              const std::vector<int64_t> &_reduction_axes,
                              const TensorInfo &_forward_op_arg_info,
                              const Op::Settings &_settings)
      : ElementWiseBinaryGradOp(_opid,
                                _reduction_axes,
                                _forward_op_arg_info,
                                _settings) {}

  std::unique_ptr<Op> clone() const final {
    return std::unique_ptr<Derived>(
        new Derived(*dynamic_cast<const Derived *>(this)));
  }

protected:
  virtual bool isArg1Grad() const final { return true; }
};

// Base class for comparison operations
class BinaryComparisonOp : public Op {
public:
  BinaryComparisonOp(const OperatorIdentifier &_opid,
                     const Op::Settings &_settings);
  void setup() final;

  // Current implementation places arg0 input at index 0, and arg1 input
  // at index 1.
  static InIndex getArg0InIndex() { return 0; }
  static InIndex getArg1InIndex() { return 1; }
  static OutIndex getOutIndex() { return 0; }

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

  bool canShard() const override { return true; }
};

} // namespace popart

#endif
