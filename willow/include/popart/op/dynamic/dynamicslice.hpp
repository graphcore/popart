// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_DYNAMICSLICE_HPP
#define GUARD_NEURALNET_DYNAMICSLICE_HPP

#include <popart/op/dynamic/dynamicbase.hpp>

namespace popart {

/**
 * Dynamic Slice Op
 *
 * This class takes two Tesnors as input (as indicated in \see
 * opidentifier.hpp)
 * 1. The Tensor to slice from
 * 2. The index of the starting point of the slice
 *    ( \see DynamicBaseOp for explanation)
 *
 * The output is the sliced tensor
 **/
class DynamicSliceOp : public DynamicSliceBaseOp {
public:
  DynamicSliceOp(const OperatorIdentifier &_opid,
                 std::vector<int64_t> axes_,
                 std::vector<int64_t> sizes_,
                 bool noOverlap_,
                 const Op::Settings &);
  std::unique_ptr<Op> clone() const final;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
};

class DynamicSlicePadGradOp : public DynamicBaseOp {
public:
  DynamicSlicePadGradOp(const OperatorIdentifier &_opid,
                        std::vector<int64_t> axes_,
                        std::vector<int64_t> sizes_,
                        bool noOverlap_,
                        const Op::Settings &settings_,
                        TensorInfo updateInInfo_ = TensorInfo());
  void setup() final;
  std::unique_ptr<Op> clone() const final;
  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const override;
  static InIndex getInIndex() { return 2; }
  std::set<InIndex> optionalInputs() const override { return {0}; }

protected:
  TensorInfo updateInInfo;
};

} // namespace popart

#endif
