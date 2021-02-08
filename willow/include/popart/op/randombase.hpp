// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_RANDOMBASE_HPP
#define GUARD_NEURALNET_RANDOMBASE_HPP

#include <popart/op/shapeorlike.hpp>

#include <memory>

namespace popart {

// Class that represents a random seed to be inserted (by the RandomSetup
// transform) as a random seed tensor into a random op. Placeholders that
// are copied or assigned are equivalent (under operator==) to one another.
// Placeholders that are constructed separately are not equivalent
// (under operator==). The RandomSetup transform will ensure that random ops
// which use equivalent placeholders will end up with the same seed input and,
// conversely, random ops that do not have equivalent placeholders will end up
// with distinct seed inputs.
class RandomSeedPlaceholder {
public:
  RandomSeedPlaceholder();
  RandomSeedPlaceholder(const RandomSeedPlaceholder &) = default;
  virtual ~RandomSeedPlaceholder()                     = default;
  RandomSeedPlaceholder &operator=(const RandomSeedPlaceholder &) = default;

private:
  // Use shared_ptr to implement the placeholder as it has the
  // properties we want under construction, copying, assignment and
  // equality tests. The value is used to implement operator< so
  // that the behaviour or RandomSeedPlaceholder is deterministic
  // when used as a key type in std::map.
  std::shared_ptr<uint64_t> placeholder;

  // Used to provide values to placeholders.
  static uint64_t placeholderCounter;

  // Friend operators.
  friend bool operator==(const RandomSeedPlaceholder &p0,
                         const RandomSeedPlaceholder &p1);
  friend bool operator!=(const RandomSeedPlaceholder &p0,
                         const RandomSeedPlaceholder &p1);
  friend bool operator<(const RandomSeedPlaceholder &p0,
                        const RandomSeedPlaceholder &p1);
};

bool operator==(const RandomSeedPlaceholder &p0,
                const RandomSeedPlaceholder &p1);
bool operator!=(const RandomSeedPlaceholder &p0,
                const RandomSeedPlaceholder &p1);
bool operator<(const RandomSeedPlaceholder &p0,
               const RandomSeedPlaceholder &p1);

// Shared base class for RNG ops
class RandomBaseOp : public ShapeOrLikeOp {
public:
  RandomBaseOp(const OperatorIdentifier &opid_,
               const OptionalDataType &dataType_,
               const Op::Settings &settings_,
               RandomSeedPlaceholder placeholder = RandomSeedPlaceholder());

  static std::vector<DataType> supportedDataTypes();

  // Call to ensure this random op will be given a random seed input that is
  // distinct from other random op's seeds.
  void useDistinctRandomSeed();
  // Call to get the random seed placeholder for this op.
  const RandomSeedPlaceholder &getRandomSeedPlaceholder() const;
  // Call to set the random seed placeholder for this op.
  void setRandomSeedPlaceholder(const RandomSeedPlaceholder &placeholder_);
  // Call to adopt the random seed placeholder from another op. Use this to
  // ensure two ops will end up using the same random seed input.
  void adoptRandomSeedPlaceholder(const RandomBaseOp &op);

  bool requiresRandomSeed() const final { return true; }

  std::vector<DataType> getSupportedDataTypes() const {
    return supportedDataTypes();
  }

  static void errorIfSeedIsSet(const Attributes &attr, OperatorIdentifier opid);

private:
  RandomSeedPlaceholder placeholder;
};

// Shared base class for RandomNormal and RandomNormalLike ops
class RandomNormalBaseOp : public RandomBaseOp {
public:
  RandomNormalBaseOp(
      const OperatorIdentifier &opid_,
      const OptionalDataType &dataType_,
      float mean_,
      float scale_,
      const Op::Settings &settings_,
      RandomSeedPlaceholder placeholder = RandomSeedPlaceholder());

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
  RandomUniformBaseOp(
      const OperatorIdentifier &opid_,
      const OptionalDataType &dataType_,
      float high_,
      float low_,
      const Op::Settings &settings_,
      RandomSeedPlaceholder placeholder = RandomSeedPlaceholder());

  void appendOutlineAttributes(OpSerialiserBase &) const final;

  float getHigh() const { return high; }
  float getLow() const { return low; }

private:
  float high;
  float low;
};

} // namespace popart

#endif
