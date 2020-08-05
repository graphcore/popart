// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_RANDOMNORMAL_HPP
#define GUARD_NEURALNET_RANDOMNORMAL_HPP

#include <vector>
#include <popart/op.hpp>

namespace popart {

class RandomNormalOp : public Op {
public:
  RandomNormalOp(const OperatorIdentifier &opid_,
                 const std::vector<int64_t> &shape_,
                 DataType dataType_,
                 float mean_,
                 float scale_,
                 const Op::Settings &settings_);

  std::unique_ptr<Op> clone() const final;

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

  void setup() final;

  static OutIndex getOutIndex() { return 0; }

  bool requiresRandomSeed() const final { return true; }
  InIndex getSeedInIndex() const final { return 0; }

  void appendOutlineAttributes(OpSerialiserBase &) const final;

  DataType getDataType() const { return dataType; }
  float getMean() const { return mean; }
  float getScale() const { return scale; }
  uint32_t getSeedModifier() const { return seedModifier; }

private:
  std::vector<int64_t> shape;
  DataType dataType;
  float mean;
  float scale;
  uint32_t seedModifier;
};

} // namespace popart

#endif
