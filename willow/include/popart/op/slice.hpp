// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_SLICE_HPP
#define GUARD_NEURALNET_SLICE_HPP

#include <popart/op.hpp>
#include <popart/op/pad.hpp>
#include <popart/slicestruct.hpp>

namespace popart {

class BaseSliceOp : public Op {

public:
  BaseSliceOp(const OperatorIdentifier &_opid,
              const std::vector<int64_t> &starts_,
              const std::vector<int64_t> &ends_,
              const std::vector<int64_t> &axes_,
              const std::vector<int64_t> &steps_,
              const Op::Settings &settings_);

  static InIndex getInIndex() { return 0; }
  static InIndex getStartsInIndex() { return 1; }
  static InIndex getEndsInIndex() { return 2; }
  static InIndex getAxesInIndex() { return 3; }
  static InIndex getStepsInIndex() { return 4; }
  static OutIndex getOutIndex() { return 0; }

  void setup() final;
  virtual void connectInTensor(InIndex, TensorId) final;

  void appendOutlineAttributes(OpSerialiserBase &) const override;

  view::RegMap fwdRegMap(InIndex, OutIndex) const final;
  view::RegMap bwdRegMap(InIndex, OutIndex) const final;
  view::Regions uses(InIndex) const final;

  view::Region createSlicedRegion(const Shape &toBeSliced) const;

  view::Region getFullInRegion() const;
  view::Region getFullOutRegion() const;

  const std::vector<int64_t> &getStarts() const { return starts; }
  const std::vector<int64_t> &getEnds() const { return ends; }
  const std::vector<int64_t> &getAxes() const { return axes; }
  const std::vector<int64_t> &getSteps() const { return steps; }
  void setStarts(const std::vector<int64_t> &x) { starts = x; }
  void setEnds(const std::vector<int64_t> &x) { ends = x; }
  void setAxes(const std::vector<int64_t> &x) { axes = x; }
  void setSteps(const std::vector<int64_t> &x) { steps = x; }

  std::vector<Slice> getSlices(std::vector<int64_t> input_shape) const;
  // assume input_shape is the shape of the input to this op:
  std::vector<Slice> getSlices() const;
  std::vector<int64_t> getPads() const;
  std::vector<unsigned> getFlips() const;

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

  // The dimension to concatentate on when unwinding the creator
  int unwindConcatDim = 0;

  bool canShard() const override { return true; }

private:
  std::vector<int64_t> starts;
  std::vector<int64_t> ends;
  std::vector<int64_t> axes;
  std::vector<int64_t> steps;

  TensorInfo createOutInfo() const;

  // In the ONNX Slice description
  // If `index > dim_size` it is treated as `index == dim_size`
  // and negative indexing is also supported.
  int64_t normalizeIndex(int64_t index, int64_t dim_size, bool flip) const;

  // if axes is empty, return default axes
  // else return axes
  static std::vector<int64_t> sanitizeAxes(const std::vector<int64_t> &starts,
                                           std::vector<int64_t> axes);

  // if steps is empty, return default steps
  // else return steps
  static std::vector<int64_t> sanitizeSteps(const std::vector<int64_t> &starts,
                                            std::vector<int64_t> steps);
};

class SliceOp : public BaseSliceOp {
public:
  SliceOp(const OperatorIdentifier &_opid,
          const std::vector<int64_t> &starts_,
          const std::vector<int64_t> &ends_,
          const std::vector<int64_t> &axes_,
          const std::vector<int64_t> &steps_,
          const Op::Settings &settings_);

  SliceOp(const OperatorIdentifier &_opid,
          const std::vector<int64_t> &starts_,
          const std::vector<int64_t> &ends_,
          const std::vector<int64_t> &axes_,
          const Op::Settings &settings_);

  std::unique_ptr<Op> clone() const override;
  std::vector<std::unique_ptr<Op>> getGradOps() final;

  std::vector<std::tuple<OperatorIdentifier, float>>
  inplacePriorityDefault() const final;

  std::unique_ptr<Op> getInplaceVariant(const OperatorIdentifier &) const final;
};

class SliceInplaceOp : public BaseSliceOp {
public:
  SliceInplaceOp(const SliceOp &);
  SliceInplaceOp(const OperatorIdentifier &_opid,
                 const std::vector<int64_t> &starts_,
                 const std::vector<int64_t> &ends_,
                 const std::vector<int64_t> &axes_,
                 const std::vector<int64_t> &steps_,
                 const Op::Settings &settings_);
  std::unique_ptr<Op> clone() const final;
  view::Regions aliases(InIndex in, OutIndex) const final;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
  // "modifies" is still the empty region
};

// The gradient of a Slice is a Pad-by-zero.
//
// Example:
//
//    1210               0
//    3235 ->  slice  -> 5 --->  ...  >--.
//    4247               7               |
//                                     loss
//    0002     slice     2               |
//    0001 <-    -    <- 1 <---  ...  <--.
//    0001     grad      1
//
// By inheriting from BasePadOutplaceOp, we make use of certain optimizations
// implemented for BasePadOutplaceOp, such as having a pre-registered Inplace Op
// (PadInplaceOp), as well as having a Pattern to convert sums of sparse
// outputs of Pads into concats where possible (see the padsum Pattern).
class SliceGradOp : public BasePadOutplaceOp {
public:
  SliceGradOp(const SliceOp &);

  void appendOutlineAttributes(OpSerialiserBase &) const override;
  std::unique_ptr<Op> clone() const final;
  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;
};

} // namespace popart

#endif
