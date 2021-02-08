// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_RANDOMNORMAL_HPP
#define GUARD_NEURALNET_RANDOMNORMAL_HPP

#include <vector>
#include <popart/op.hpp>
#include <popart/op/randombase.hpp>

namespace popart {

class RandomNormalOp : public RandomNormalBaseOp {
public:
  RandomNormalOp(const OperatorIdentifier &opid_,
                 const std::vector<int64_t> &shape_,
                 const OptionalDataType &dataType_,
                 float mean_,
                 float scale_,
                 const Op::Settings &settings_,
                 RandomSeedPlaceholder placeholder = RandomSeedPlaceholder());

  std::unique_ptr<Op> clone() const final;
  void setup() final;
  InIndex getSeedInIndex() const final { return 0; }

private:
  std::vector<int64_t> shape;
};

class RandomNormalLikeOp : public RandomNormalBaseOp {
public:
  RandomNormalLikeOp(
      const OperatorIdentifier &opid_,
      const OptionalDataType &dataType_,
      float mean_,
      float scale_,
      const Op::Settings &settings_,
      RandomSeedPlaceholder placeholder = RandomSeedPlaceholder());

  std::unique_ptr<Op> clone() const final;
  void setup() final;
  std::vector<std::unique_ptr<Op>> getGradOps() final;

  static InIndex getInIndex() { return 0; }
  InIndex getSeedInIndex() const final { return 1; }

  std::unique_ptr<RandomNormalOp> foldInputTensor(const Op::Settings &) const;
};

} // namespace popart

#endif
