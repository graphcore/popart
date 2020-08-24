// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_RANDOMBASE_HPP
#define GUARD_NEURALNET_RANDOMBASE_HPP

#include <vector>
#include <popart/op.hpp>

namespace popart {

// Shared base class for RNG ops
class RandomBaseOp : public Op {
public:
  RandomBaseOp(const OperatorIdentifier &opid_,
               const OptionalDataType &dataType_,
               const Op::Settings &settings_);

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

  static OutIndex getOutIndex() { return 0; }

  bool requiresRandomSeed() const final { return true; }

  uint32_t getSeedModifier() const { return seedModifier; }

  static void validateDataType(DataType dataType, OperatorIdentifier opid);

  static std::vector<DataType> getSupportedDataTypes();

  static OptionalDataType getOptionalDataType(const Attributes &attr,
                                              OperatorIdentifier opid);

  static void errorIfSeedIsSet(const Attributes &attr, OperatorIdentifier opid);

protected:
  const OptionalDataType &getDataType() const { return dataType; }

  void setupWithShape(const std::vector<int64_t> &shape);

  void setupLike(const popart::TensorInfo &info);

private:
  OptionalDataType dataType;
  uint32_t seedModifier;
};

// Shared base class for RandomNormal and RandomNormalLike ops
class RandomNormalBaseOp : public RandomBaseOp {
public:
  RandomNormalBaseOp(const OperatorIdentifier &opid_,
                     const OptionalDataType &dataType_,
                     float mean_,
                     float scale_,
                     const Op::Settings &settings_);

  void appendOutlineAttributes(OpSerialiserBase &) const final;

  float getMean() const { return mean; }
  float getScale() const { return scale; }

private:
  float mean;
  float scale;
};

// Shared base class for RandomUniform and RandomUniformLike ops
class RandomUniformBaseOp : public RandomBaseOp {
public:
  RandomUniformBaseOp(const OperatorIdentifier &opid_,
                      const OptionalDataType &dataType_,
                      float high_,
                      float low_,
                      const Op::Settings &settings_);

  void appendOutlineAttributes(OpSerialiserBase &) const final;

  float getHigh() const { return high; }
  float getLow() const { return low; }

private:
  float high;
  float low;
};

} // namespace popart

#endif
