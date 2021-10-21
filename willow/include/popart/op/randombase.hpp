// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_RANDOMBASE_HPP
#define GUARD_NEURALNET_RANDOMBASE_HPP

#include <popart/op/shapeorlike.hpp>

#include <memory>

namespace popart {

// Shared base class for RNG ops
class RandomBaseOp : public ShapeOrLikeOp {
public:
  RandomBaseOp(const OperatorIdentifier &opid_,
               const OptionalDataType &dataType_,
               const Op::Settings &settings_);

  static std::vector<DataType> supportedDataTypes();

  bool requiresRandomSeed() const final { return true; }

  std::vector<DataType> getSupportedDataTypes() const {
    return supportedDataTypes();
  }

  static void errorIfSeedIsSet(const Attributes &attr, OperatorIdentifier opid);
};

// Shared base class for RandomNormal and RandomNormalLike ops
class RandomNormalBaseOp : public RandomBaseOp {
public:
  RandomNormalBaseOp(const OperatorIdentifier &opid_,
                     const OptionalDataType &dataType_,
                     float mean_,
                     float scale_,
                     const Op::Settings &settings_);

  void appendOutlineAttributes(OpSerialiserBase &) const override;

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

  void appendOutlineAttributes(OpSerialiserBase &) const override;

  float getHigh() const { return high; }
  float getLow() const { return low; }

private:
  float high;
  float low;
};

} // namespace popart

#endif
