// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_RANDOMUNIFORM_HPP
#define GUARD_NEURALNET_RANDOMUNIFORM_HPP

#include <vector>
#include <popart/op.hpp>

namespace popart {

class RandomUniformOp : public Op {
public:
  RandomUniformOp(const OperatorIdentifier &opid_,
                  const std::vector<int64_t> &shape_,
                  DataType dataType_,
                  float high_,
                  float low_,
                  const Op::Settings &settings_);

  std::unique_ptr<Op> clone() const final;

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

  void setup() final;

  static OutIndex getOutIndex() { return 0; }

  bool requiresRandomSeed() const final { return true; }
  InIndex getSeedInIndex() const final { return 0; }

  void appendOutlineAttributes(OpSerialiserBase &) const final;

  DataType getDataType() const { return dataType; }
  float getHigh() const { return high; }
  float getLow() const { return low; }
  uint32_t getSeedModifier() const { return seedModifier; }

private:
  std::vector<int64_t> shape;
  DataType dataType;
  float high;
  float low;
  uint32_t seedModifier;
};

} // namespace popart

#endif
