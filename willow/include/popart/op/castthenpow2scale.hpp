// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OP_CASTTHENPOW2SCALE_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_CASTTHENPOW2SCALE_HPP_

#include <memory>
#include <vector>
#include <popart/op.hpp>

#include "popart/datatype.hpp"
#include "popart/names.hpp"

namespace popart {
struct OperatorIdentifier;

/**
 * A cast operation from floating point 8 to float 16 or float 32.
 *
 * This is distinct from other cast operations as it requires a INT8 scaling
 * tensor to be passed at input index 1. See the section on floating point 8
 * data types in the PopART user guide for details about this operator.
 *
 * \param _opid The operator identifier to use.
 * \param _to DataType to cast to, FLOAT or FLOAT16.
 * \param settings Operator settings to use.
 */
class CastThenPow2ScaleOp : public Op {
public:
  CastThenPow2ScaleOp(const OperatorIdentifier &_opid,
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
   * \returns DataType FLOAT or FLOAT16 data type.
   */
  DataType toDataType() const { return to; }

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

  bool canShard() const override { return true; }

  ReplicatedTensorShardingIndices
  getReplicatedTensorShardingIndices() const override {
    return {{{CastThenPow2ScaleOp::getInIndex()},
             {CastThenPow2ScaleOp::getOutIndex()}}};
  }

  bool canBeReplacedByIdentity() const override;

private:
  DataType to;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_OP_CASTTHENPOW2SCALE_HPP_
