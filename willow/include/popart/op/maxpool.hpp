// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_MAXPOOL_HPP
#define GUARD_NEURALNET_MAXPOOL_HPP

#include <cstdint>
#include <map>
#include <memory>
#include <vector>
#include <popart/names.hpp>
#include <popart/op.hpp>
#include <popart/op/receptive.hpp>

#include "popart/tensorinfo.hpp"

namespace popart {
class OpSerialiserBase;
struct OperatorIdentifier;

// c++ note : the conditions are suitable here
// for the compiler to generate defaults for
// "the 3": destructor, copy constructor, assigment op.
class MaxPoolOp : public HasReceptiveFieldOp {
public:
  MaxPoolOp(const OperatorIdentifier &_opid,
            const std::vector<int64_t> &kernelShape_,
            int64_t storageOrder,
            const HasReceptiveFieldOp::ReceptiveOpAttributes &attributes,
            const Op::Settings &settings);
  std::unique_ptr<Op> clone() const final;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
  int64_t getNOutChans() const final;

  static InIndex getInIndex() { return 0; }
  static OutIndex getOutIndex() { return 0; }

  void appendOutlineAttributes(OpSerialiserBase &) const override;

  float getSubgraphValue() const final { return getLowSubgraphValue(); }
  bool canBeReplacedByIdentity() const override;

  Shape getSpatialK() const final;

private:
  void setup0() const final;

  int64_t storageOrder;
  std::vector<int64_t> kernelShape;
};

class MaxPoolGradOp : public Op {
public:
  MaxPoolGradOp(const MaxPoolOp &);
  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;
  void setup() final;
  std::unique_ptr<Op> clone() const final;

  static InIndex getPrePooledInIndex() { return 0; }
  static InIndex getPooledInIndex() { return 1; }
  static InIndex getGradPooledInIndex() { return 2; }
  static OutIndex getOutIndex() { return 0; }

  void appendOutlineAttributes(OpSerialiserBase &) const override;

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

private:
  // The shape and type of the input to the
  // forward op which creates this backwards op
  TensorInfo unpooledInfo;

public:
  const Shape creatorSpatialK;
  const Shape creatorStrides;
  const Shape creatorLowerPads;
  const Shape creatorUpperPads;
};

} // namespace popart

#endif
