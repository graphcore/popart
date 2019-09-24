#ifndef GUARD_NEURALNET_SLICE_HPP
#define GUARD_NEURALNET_SLICE_HPP

#include <popart/op.hpp>

namespace popart {

struct Slice {
  int64_t start;
  int64_t end;
  int64_t axis;

  Slice(int64_t start_, int64_t end_, int64_t axis_);
};

class BaseSliceOp : public Op {

public:
  BaseSliceOp(const OperatorIdentifier &_opid,
              const std::vector<int64_t> &starts_,
              const std::vector<int64_t> &ends_,
              const std::vector<int64_t> &axes_,
              const Op::Settings &settings_);

  static InIndex getInIndex() { return 0; }
  static InIndex getStartsInIndex() { return 1; }
  static InIndex getEndsInIndex() { return 2; }
  static InIndex getAxesInIndex() { return 3; }
  static InIndex getStepsInIndex() { return 4; }
  static OutIndex getOutIndex() { return 0; }

  void setup() final;
  virtual void connectInTensor(InIndex, TensorId) final;

  void appendAttributes(OpSerialiserBase &) const override;

  view::RegMap fwdRegMap(InIndex) const final;
  view::RegMap bwdRegMap(OutIndex) const final;
  view::Region uses(InIndex) const final;

  view::Region createSlicedRegion(const Shape &toBeSliced) const;

  view::Region getFullInRegion() const;
  view::Region getFullOutRegion() const;

  const std::vector<int64_t> &getStarts() const { return starts; }
  const std::vector<int64_t> &getEnds() const { return ends; }
  const std::vector<int64_t> &getAxes() const { return axes; }
  void setStarts(const std::vector<int64_t> &x) { starts = x; }
  void setEnds(const std::vector<int64_t> &x) { ends = x; }
  void setAxes(const std::vector<int64_t> &x) { axes = x; }

  std::vector<Slice> getSlices(std::vector<int64_t> input_shape) const;
  // assume input_shape is the shape of the input to this op:
  std::vector<Slice> getSlices() const;

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

  // The dimension to concatentate on when unwinding the creator
  int unwindConcatDim = 0;

  // The set of all slices of the input.
  // Used by the mat mul serialization transform so that we can unwind a creator
  std::vector<TensorId> allSlices;

private:
  std::vector<int64_t> starts;
  std::vector<int64_t> ends;
  std::vector<int64_t> axes;

  TensorInfo createOutShape() const;

  // In the ONNX Slice description
  // If `index > dim_size` it is treated as `index == dim_size`
  // and negative indexing is also supported.
  int64_t normalizeIndex(int64_t index, int64_t dim_size) const;

  // if axes is empty, return default axes
  // else return axes
  static std::vector<int64_t> sanitizeAxes(const std::vector<int64_t> &starts,
                                           std::vector<int64_t> axes);
};

class SliceOp : public BaseSliceOp {
public:
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
                 const Op::Settings &settings_);
  std::unique_ptr<Op> clone() const final;
  view::Region aliases(InIndex) const final;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
  // "modifies" is still the empty region
};

} // namespace popart

#endif
