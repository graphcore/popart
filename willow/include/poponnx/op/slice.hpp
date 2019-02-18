#ifndef GUARD_NEURALNET_SLICE_HPP
#define GUARD_NEURALNET_SLICE_HPP

#include <poponnx/op.hpp>

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
          const std::vector<int64_t> &starts_,
          const std::vector<int64_t> &ends_,
          const std::vector<int64_t> &axes_,
          const Op::Settings &settings_);
  std::unique_ptr<Op> clone() const override;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
  void setup() final;

  static InIndex getInIndex() { return 0; }
  static OutIndex getOutIndex() { return 0; }

  std::vector<Slice> getSlices() const;

  void appendAttributes(std::stringstream &ss,
                        const std::string &tab) const override;

private:
  const std::vector<int64_t> starts;
  const std::vector<int64_t> ends;
  const std::vector<int64_t> axes;

  TensorInfo createOutShape() const;

  // In the ONNX Slice description
  // If `index > dim_size` it is treated as `index == dim_size`
  // and negative indexing is also supported.
  static int64_t normalizeIndex(int64_t index, int64_t dim_size);

  // if axes is empty, return default axes
  // else return axes
  static std::vector<int64_t> sanitizeAxes(const std::vector<int64_t> starts,
                                           std::vector<int64_t> axes);
};

} // namespace poponnx

#endif
