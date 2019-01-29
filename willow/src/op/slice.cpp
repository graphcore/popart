#include <poponnx/makeunique.hpp>
#include <poponnx/op/pad.hpp>
#include <poponnx/op/slice.hpp>
#include <poponnx/op/slicegrad.hpp>
#include <poponnx/opmanager.hpp>

namespace poponnx {

Slice::Slice(int64_t start_, int64_t end_, int64_t axis_)
    : start(start_), end(end_), axis(axis_) {}

SliceOp::SliceOp(const OperatorIdentifier &_opid,
                 const std::vector<int64_t> &starts_,
                 const std::vector<int64_t> &ends_,
                 const std::vector<int64_t> &axes_,
                 const Op::Settings &settings_)
    : Op(_opid, settings_), starts(starts_), ends(ends_), axes(axes_) {

  if (axes.size() == 0) {
    for (int i = 0; i < starts.size(); i++) {
      axes.push_back(i);
    }
  }
}

std::unique_ptr<Op> SliceOp::clone() const {
  return make_unique<SliceOp>(*this);
}

std::vector<std::unique_ptr<Op>> SliceOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(make_unique<SliceGradOp>(*this));
  return upops;
}

void SliceOp::setup() {
  auto input_shape = inShape(getInIndex());

  for (auto slice : getSlices()) {
    auto new_size           = slice.end - slice.start;
    input_shape[slice.axis] = new_size;
  }

  outInfo(getOutIndex()) = {inInfo(getInIndex()).dataType(), input_shape};
}

std::vector<Slice> SliceOp::getSlices() const {
  std::vector<Slice> slices;

  auto input_shape = inShape(getInIndex());

  for (int i = 0; i < axes.size(); i++) {
    auto axis     = axes[i];
    auto dim_size = input_shape[axis];
    auto begin    = normalizeIndex(starts[i], dim_size);
    auto end      = normalizeIndex(ends[i], dim_size);

    slices.emplace_back(begin, end, axis);
  }

  return slices;
}

// In the ONNX Slice Operator
// If `index > dim_size` it is treated as `index == dim_size`
// and negative indexing is also supported.
int64_t SliceOp::normalizeIndex(int64_t index, int64_t dim_size) {
  index = std::min(index, dim_size);

  if (index < 0) {
    if (dim_size + index < 0) {
      throw error(
          "index {} is out of bounds for axis with size {}", index, dim_size);
    }

    index = dim_size + index;
  }

  return index;
}

void SliceOp::appendAttributes(std::stringstream &ss,
                               const std::string &tab) const {
  Op::appendAttributes(ss, tab);
  appendAttribute(ss, tab, "starts", starts);
  appendAttribute(ss, tab, "ends", ends);
  appendAttribute(ss, tab, "axes", axes);
}

SliceGradOp::SliceGradOp(const SliceOp &op_)
    : PadOp(Onnx::GradOperators::SliceGrad,
            calculatePadding(op_),
            0,
            "constant",
            op_.getSettings()) {}

const std::vector<GradInOutMapper> &SliceGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {getInIndex(), SliceOp::getOutIndex(), GradOpInType::GRADOUT}};
  return inInfo;
}

const std::map<int, int> &SliceGradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {
      {getOutIndex(), SliceOp::getInIndex()}};
  return outInfo;
}

std::vector<int64_t> SliceGradOp::calculatePadding(const SliceOp &slice_op) {
  auto t_rank   = slice_op.inInfo(SliceOp::getInIndex()).rank();
  auto in_shape = slice_op.inInfo(SliceOp::getInIndex()).shape();
  std::vector<int64_t> pads(t_rank * 2, 0);

  for (auto slice : slice_op.getSlices()) {
    pads[slice.axis]          = slice.start;
    pads[slice.axis + t_rank] = in_shape[slice.axis] - slice.end;
  }

  return pads;
}

namespace {
static OpCreator<SliceOp>
    sliceOpCreator(Onnx::Operators::Slice_1,
                   [](const OperatorIdentifier &_opid,
                      const Op::Settings &settings,
                      const Attributes &attr) -> std::unique_ptr<Op> {
                     std::vector<int64_t> starts =
                         attr.getAttribute<Attributes::Ints>("starts", {});
                     std::vector<int64_t> ends =
                         attr.getAttribute<Attributes::Ints>("ends", {});
                     std::vector<int64_t> axes =
                         attr.getAttribute<Attributes::Ints>("axes", {});

                     return std::unique_ptr<Op>(
                         new SliceOp(_opid, starts, ends, axes, settings));
                   },
                   true);
} // namespace

} // namespace poponnx
