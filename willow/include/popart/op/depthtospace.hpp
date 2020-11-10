// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_DEPTHTOSPACE_HPP
#define GUARD_NEURALNET_DEPTHTOSPACE_HPP

#include <popart/op.hpp>

namespace popart {

// DCR mode: elements along the depth dimension from the input tensor
// are rearranged in the following order: depth, column, and then row.
// CRD mode: elements along the depth dimension from the input tensor
// are rearranged in the following order: column, row, and the depth.
enum class DepthToSpaceMode { DCR, CRD };
std::string toString(DepthToSpaceMode);
std::ostream &operator<<(std::ostream &, DepthToSpaceMode);

class DepthToSpaceBaseOp : public Op {
public:
  DepthToSpaceBaseOp(const OperatorIdentifier &_opid,
                     int64_t blocksize_,
                     DepthToSpaceMode mode_,
                     const Op::Settings &settings_);

  void setup() final;
  std::vector<std::unique_ptr<Op>> getGradOps() final;

  int64_t getBlocksize() const { return blocksize; }
  DepthToSpaceMode getMode() const { return mode; }

  static InIndex getInIndex() { return 0; }
  static OutIndex getOutIndex() { return 0; }

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

private:
  const int64_t blocksize;
  const DepthToSpaceMode mode;
};

// DepthToSpace op rearranges (permutes) data from depth into
// blocks of spatial data. It outputs a copy of the input tensor
// where values from the depth dimension are moved in spatial blocks
// to the height and width dimensions.
// It expect 4 dimensional input tensor: x.shape = b, c, h, w. See
// https://github.com/onnx/onnx/blob/master/docs/Operators.md#DepthToSpace
class DepthToSpaceOp : public DepthToSpaceBaseOp {
public:
  DepthToSpaceOp(const OperatorIdentifier &_opid,
                 int64_t blocksize_,
                 DepthToSpaceMode mode_,
                 const Op::Settings &settings_);
  std::unique_ptr<Op> clone() const final;
};

} // namespace popart

#endif
