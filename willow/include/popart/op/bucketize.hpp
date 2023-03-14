// Copyright (c) 2023 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OP_BUCKETIZE_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_BUCKETIZE_HPP_

#include <memory>
#include <popart/op.hpp>

namespace popart {

class BucketizeOp : public Op {
public:
  BucketizeOp(const OperatorIdentifier &opid,
              bool right,
              const Op::Settings &settings);

  static InIndex inIndex() { return 0; }
  static InIndex boundariesInIndex() { return 1; }
  static OutIndex outIndex() { return 0; }

  void setup() override;
  std::unique_ptr<Op> clone() const override;
  float getSubgraphValue() const override;
  void appendOutlineAttributes(OpSerialiserBase &) const override;

  bool isRight() const noexcept;

private:
  bool right_;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_OP_BUCKETIZE_HPP_
