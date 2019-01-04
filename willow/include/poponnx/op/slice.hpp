#ifndef GUARD_NEURALNET_SLICE_HPP
#define GUARD_NEURALNET_SLICE_HPP

#include <poponnx/op/pad.hpp>

namespace poponnx {

struct Slice {
  int64_t start;
  int64_t end;
  int64_t axis;

  Slice(int64_t start_, int64_t end_, int64_t axis_);
};

class SliceOp : public Op {
public:
  SliceOp(const OperatorIdentifier &_opid,
          Ir *_ir,
          const std::string &name = "",
          const Attributes &_attr = {});
  std::unique_ptr<Op> clone() const final;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
  void setup() final;

  static InIndex getInIndex() { return 0; }
  static OutIndex getOutIndex() { return 0; }

  std::vector<Slice> getSlices() const;

private:
  std::vector<int64_t> axes;
  std::vector<int64_t> starts;
  std::vector<int64_t> ends;

  // In the ONNX Slice Operator
  // If `index > dim_size` it is treated as `index == dim_size`
  // and negative indexing is also supported.
  static int64_t normalizeIndex(int64_t index, int64_t dim_size);
};

class SliceGradOp : public PadOp {
public:
  SliceGradOp(SliceOp *);

  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;

private:
  static std::vector<int64_t> calculatePadding(SliceOp *);
};

} // namespace poponnx

#endif
