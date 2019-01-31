#include <poponnx/makeunique.hpp>
#include <poponnx/op/pad.hpp>
#include <poponnx/op/slice.hpp>
#include <poponnx/op/slicegrad.hpp>
#include <poponnx/opmanager.hpp>

namespace poponnx {

Slice::Slice(int64_t start_, int64_t end_, int64_t axis_)
    : start(start_), end(end_), axis(axis_) {}

SliceImpl::SliceImpl(const std::vector<int64_t> &starts_,
                     const std::vector<int64_t> &ends_,
                     const std::vector<int64_t> &axes_)
    : starts(starts_), ends(ends_), axes(axes_) {
  if (axes.size() == 0) {
    for (int i = 0; i < starts.size(); i++) {
      axes.push_back(i);
    }
  }
}

TensorInfo SliceImpl::createOutShape(const TensorInfo &in_info) const {
  auto output_shape = in_info.shape();

  for (auto slice : getSlices(in_info.shape())) {
    auto new_size            = slice.end - slice.start;
    output_shape[slice.axis] = new_size;
  }

  return {in_info.dataType(), output_shape};
}

std::vector<Slice>
SliceImpl::getSlices(std::vector<int64_t> input_shape) const {
  std::vector<Slice> slices;
  slices.reserve(axes.size());

  for (int i = 0; i < axes.size(); i++) {
    auto axis     = axes[i];
    auto dim_size = input_shape[axis];
    auto begin    = normalizeIndex(starts[i], dim_size);
    auto end      = normalizeIndex(ends[i], dim_size);

    slices.emplace_back(begin, end, axis);
  }

  return slices;
}

// In the ONNX Slice Implerator
// If `index > dim_size` it is treated as `index == dim_size`
// and negative indexing is also supported.
int64_t SliceImpl::normalizeIndex(int64_t index, int64_t dim_size) {
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

const std::vector<int64_t> &SliceImpl::getStarts() const { return starts; }
const std::vector<int64_t> &SliceImpl::getEnds() const { return ends; }
const std::vector<int64_t> &SliceImpl::getAxes() const { return axes; }

SliceOp::SliceOp(const OperatorIdentifier &_opid,
                 const std::vector<int64_t> &starts_,
                 const std::vector<int64_t> &ends_,
                 const std::vector<int64_t> &axes_,
                 const Op::Settings &settings_)
    : Op(_opid, settings_), impl(starts_, ends_, axes_) {}

std::unique_ptr<Op> SliceOp::clone() const {
  return make_unique<SliceOp>(*this);
}

std::vector<std::unique_ptr<Op>> SliceOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(make_unique<SliceGradOp>(*this));
  return upops;
}

void SliceOp::setup() {
  outInfo(getOutIndex()) = impl.createOutShape(inInfo(getInIndex()));
}

std::vector<Slice> SliceOp::getSlices() const {
  return impl.getSlices(inShape(getInIndex()));
}

void SliceOp::appendAttributes(std::stringstream &ss,
                               const std::string &tab) const {
  Op::appendAttributes(ss, tab);
  appendAttribute(ss, tab, "starts", impl.getStarts());
  appendAttribute(ss, tab, "ends", impl.getEnds());
  appendAttribute(ss, tab, "axes", impl.getAxes());
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
