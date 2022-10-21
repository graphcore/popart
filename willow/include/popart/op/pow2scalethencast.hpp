// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OP_POW2SCALETHENCAST_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_POW2SCALETHENCAST_HPP_

#include <memory>
#include <vector>
#include <popart/op.hpp>

#include "popart/datatype.hpp"
#include "popart/names.hpp"

namespace popart {
struct OperatorIdentifier;

/**
 * A cast to floating point 8 op
 *
 * This is distinct from other cast operations as it requires a INT8 scaling
 * tensor to be passed at input index 1. See the section on floating point 8
 * data types in the PopART user guide for details about this operator.
 *
 * \param _opid The operator identifier to use.
 * \param _to Either "FLOAT8_143" or "FLOAT8_152".
 * \param settings Operator settings to use.
 */
class Pow2ScaleThenCastOp : public Op {
public:
  Pow2ScaleThenCastOp(const OperatorIdentifier &_opid,
                      const DataType _to,
                      const Op::Settings &settings);
  std::unique_ptr<Op> clone() const override;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
  void setup() override;

  static InIndex getInIndex() { return 0; }
  static InIndex getlog2ScaleInIndex() { return 1; }
  static OutIndex getOutIndex() { return 0; }

  /**
   * Get the datatype to cast to.
   *
   * \returns DataType FLOAT8_143 or FLOAT8_152 data type.
   */
  DataType toDataType() const { return to; }

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

  bool canShard() const override { return true; }

  ReplicatedTensorShardingIndices
  getReplicatedTensorShardingIndices() const override {
    return {{{Pow2ScaleThenCastOp::getInIndex()},
             {Pow2ScaleThenCastOp::getOutIndex()}}};
  }

  bool canBeReplacedByIdentity() const override;

private:
  DataType to;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_OP_POW2SCALETHENCAST_HPP_
