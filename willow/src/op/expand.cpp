#include <popart/broadcastutil.hpp>
#include <popart/op/expand.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/region.hpp>
#include <popart/tensor.hpp>
namespace popart {

ExpandInplaceOp::ExpandInplaceOp(const OperatorIdentifier &_opid,
                                 const Shape &_outShape,
                                 const Op::Settings &settings_)
    : ExpandOp(_opid, _outShape, settings_) {}

ExpandOp::ExpandOp(const OperatorIdentifier &_opid,
                   const Shape &_outShape,
                   const Op::Settings &_settings)
    : Op(_opid, _settings), outShape(_outShape) {}

ExpandInplaceOp::ExpandInplaceOp(const ExpandOp &op)
    : ExpandOp(Onnx::CustomOperators::ExpandInplace,
               op.getOutShape(),
               op.getSettings()) {}

void ExpandOp::regMapPreChecks(InIndex inIndex) const {
  if (inIndex >= input->tensorMap().size() || inIndex < 0) {
    throw error("invalid index in ExpandOp::fwdRegMap");
  }
}

view::RegMap ExpandOp::fwdRegMap(InIndex inIndex, OutIndex) const {
  regMapPreChecks(inIndex);

  auto out_shape = getOutShape();
  auto in_shape  = input->getIndexShapeMap()[inIndex];
  return [out_shape, in_shape](const view::Region &r) {
    auto out_size  = static_cast<int>(out_shape.size());
    auto arg_shape = padShape(in_shape, out_size, int64_t{1});
    auto lower     = padShape(r.getLower(), out_size, int64_t{0});
    auto upper     = padShape(r.getUpper(), out_size, int64_t{1});

    if (r.isEmpty()) {
      return view::Regions(1, view::Region::getEmpty(out_shape.size()));
    }

    // broadcasting
    for (int i = 0; i < out_shape.size(); i++) {
      if (arg_shape[i] == 1 && out_shape[i] > 1) {
        upper[i] = out_shape[i];
      }
    }

    return view::Regions(1, view::Region{lower, upper});
  };
}

view::RegMap ExpandOp::bwdRegMap(InIndex inIndex, OutIndex) const {
  regMapPreChecks(inIndex);
  auto out_shape = getOutShape();
  auto in_shape  = input->getIndexShapeMap()[inIndex];
  auto in_size   = static_cast<int>(in_shape.size());
  auto upper     = unpadShape(out_shape, in_size);
  return
      [out_shape, in_shape, in_size, upper](const view::Region &r_out) mutable {
        auto size_diff = r_out.getLower().size() - in_size;
        auto lower     = unpadShape(r_out.getLower(), in_size);
        if (r_out.isEmpty()) {
          return view::Regions(1, view::Region::getEmpty(out_shape.size()));
        }
        if (lower.size() > 0) { // handling scalar input
          for (size_t i = size_diff; i < out_shape.size(); i++) {
            if (in_shape[i - size_diff] != out_shape[i]) {
              lower[i - size_diff] = 0;
              upper[i - size_diff] = 1;
            }
          }
        }
        // TODO T8446 : check intersect?
        return view::Regions(1, view::Region(lower, upper));
      };
}

std::unique_ptr<Op>
ExpandOp::getInplaceVariant(const OperatorIdentifier &operator_id) const {
  if (operator_id == Onnx::CustomOperators::ExpandInplace) {
    return std::make_unique<ExpandInplaceOp>(*this);
  }

  return Op::getInplaceVariant(operator_id);
}

void ExpandOp::setup() {
  outInfo(getOutIndex()) = TensorInfo(
      input->tensor(ExpandOp::getInTensorIndex())->info.data_type(), outShape);
}

std::vector<std::unique_ptr<Op>> ExpandOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> result;
  result.push_back(std::make_unique<ExpandGradOp>(*this));

  return result;
}

void ExpandOp::connectInTensor(InIndex inIndex, TensorId tenId) {
  // index 0 is the data tensor to be expanded. We connect
  // the data tensor to this Op as an input, the default connection of
  // an input tensor to its Op
  if (inIndex == 0) {
    Op::connectInTensor(inIndex, tenId);
  } else if (inIndex == 1) {
    // we attempt to set outputInfo
    try {
      getInTensorData(tenId, outShape);
      finaliseShape();
    } catch (popart::error &err) {
      throw error("Need the value of the {} input 'shape' to detemine the "
                  "output shape, but was unable because {}",
                  opid,
                  err.what());
    }

  } else {
    throw error("Unexpected index " + std::to_string(inIndex) +
                " in ExpandOp::connectInTensor");
  }
}

void ExpandOp::finaliseShape() {
  auto input_shape = inShape(ExpandOp::getInTensorIndex());
  for (int64_t i = outShape.size() - 1, j = input_shape.size() - 1;
       i >= 0 || j >= 0;
       --i, --j) {
    const size_t shape_x = (j >= 0 ? input_shape[j] : 1);
    const size_t shape_y = (i >= 0 ? outShape[i] : 1);
    if (shape_x != 1 && shape_y != 1 && shape_x != shape_y) {
      throw error("Expand shape constraint: corresponding dimensions must have "
                  "the same value or one of them must be 1");
    }
    outShape[i] = std::max(shape_x, shape_y);
  }
}

ExpandGradOp::ExpandGradOp(const ExpandOp &fwd)
    : Op(Onnx::GradOperators::ExpandGrad, fwd.getSettings()) {
  fwdInput               = ExpandOp::getInTensorIndex();
  const DataType outType = fwd.inInfo(ExpandOp::getInTensorIndex()).dataType();
  Shape outShape         = fwd.inShape(ExpandOp::getInTensorIndex());

  gradInfo               = TensorInfo(outType, outShape);
  gradOutToNonGradInInfo = {{getOutIndex(), ExpandOp::getInTensorIndex()}};
}

ExpandGradOp::ExpandGradOp(const OperatorIdentifier &_opid,
                           const ExpandGradOp &expand_grad_op)
    : Op(_opid, expand_grad_op.getSettings()),
      fwdInput(expand_grad_op.fwdInput), gradInfo(expand_grad_op.gradInfo),
      gradOutToNonGradInInfo(expand_grad_op.gradOutToNonGradInInfo) {}

std::unique_ptr<Op> ExpandGradOp::clone() const {
  return std::make_unique<ExpandGradOp>(*this);
}

const std::vector<GradInOutMapper> &ExpandGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {getDYIndex(), ExpandOp::getOutIndex(), GradOpInType::GRADOUT}};
  return inInfo;
}

const std::map<int, int> &ExpandGradOp::gradOutToNonGradIn() const {
  return gradOutToNonGradInInfo;
}

void ExpandGradOp::setup() { outInfo(getOutIndex()) = gradInfo; }

namespace {

static OpDefinition::DataTypes T  = {DataType::UINT8,
                                    DataType::UINT16,
                                    DataType::UINT32,
                                    DataType::UINT64,
                                    DataType::INT8,
                                    DataType::INT16,
                                    DataType::INT32,
                                    DataType::INT64,
                                    DataType::FLOAT16,
                                    DataType::FLOAT,
                                    DataType::BOOL};
static OpDefinition::DataTypes T1 = {DataType::INT64};

static OpDefinition
    expandOpDef({OpDefinition::Inputs({{"input", T}, {"out_shape", T1, true}}),
                 OpDefinition::Outputs({{"expand_result", T}}),
                 OpDefinition::Attributes()});

static OpCreator<ExpandOp>
    expandOpCreator(OpDefinitions({{Onnx::Operators::Expand_8, expandOpDef}}));
} // namespace

} // namespace popart
