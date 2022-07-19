// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OP_RANDOMNORMAL_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_RANDOMNORMAL_HPP_

#include <cstdint>
#include <memory>
#include <vector>
#include <popart/op.hpp>
#include <popart/op/randombase.hpp>

#include "popart/datatype.hpp"
#include "popart/names.hpp"

namespace popart {
struct OperatorIdentifier;

class RandomNormalOp : public RandomNormalBaseOp {
public:
  RandomNormalOp(const OperatorIdentifier &opid_,
                 const Shape &shape_,
                 const OptionalDataType &dataType_,
                 float mean_,
                 float scale_,
                 const Op::Settings &settings_);

  std::unique_ptr<Op> clone() const final;
  void setup() final;
  InIndex getSeedInIndex() const final { return 0; }

private:
  std::vector<int64_t> shape;
};

class RandomNormalLikeOp : public RandomNormalBaseOp {
public:
  RandomNormalLikeOp(const OperatorIdentifier &opid_,
                     const OptionalDataType &dataType_,
                     float mean_,
                     float scale_,
                     const Op::Settings &settings_);

  std::unique_ptr<Op> clone() const final;
  void setup() final;
  std::vector<std::unique_ptr<Op>> getGradOps() final;

  static InIndex getInIndex() { return 0; }
  InIndex getSeedInIndex() const final { return 1; }

  std::unique_ptr<RandomNormalOp> foldInputTensor(const Op::Settings &) const;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_OP_RANDOMNORMAL_HPP_
