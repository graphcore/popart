#include <memory>
#include <popart/onnxutil.hpp>
#include <popart/op/cast.hpp>
#include <popart/opmanager.hpp>

namespace popart {

CastOp::CastOp(const OperatorIdentifier &_opid,
               DataType _to,
               const Op::Settings &settings_)
    : Op(_opid, settings_), to(_to) {}

std::unique_ptr<Op> CastOp::clone() const {
  return std::make_unique<CastOp>(*this);
}

std::vector<std::unique_ptr<Op>> CastOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(std::make_unique<CastGradOp>(*this));
  return upops;
}

void CastOp::setup() { outInfo(getOutIndex()) = {to, inShape(getInIndex())}; }

CastGradOp::CastGradOp(const CastOp &fwdOp)
    : CastOp(Onnx::GradOperators::CastGrad,
             fwdOp.inInfo(getInIndex()).dataType(),
             fwdOp.getSettings()) {}

std::unique_ptr<Op> CastGradOp::clone() const {
  return std::make_unique<CastGradOp>(*this);
}

const std::vector<GradInOutMapper> &CastGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {getInIndex(), CastOp::getOutIndex(), GradOpInType::GRADOUT}};

  return inInfo;
}

const std::map<int, int> &CastGradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {
      {getOutIndex(), CastOp::getInIndex()}};

  return outInfo;
}

namespace {
static OpCreator<CastOp> castOpCreator(
    {Onnx::Operators::Cast_1, Onnx::Operators::Cast_6, Onnx::Operators::Cast_9},
    [](const OperatorIdentifier &opid,
       const Op::Settings &settings,
       const Attributes &attr) -> std::unique_ptr<Op> {
      int64_t i64_to;
      attr.set(i64_to, "to");
      auto tpdt_to   = static_cast<onnx::TensorProto_DataType>(i64_to);
      DataType dt_to = onnxutil::getDataType(tpdt_to);

      return std::make_unique<CastOp>(opid, dt_to, settings);
    },
    true);
} // namespace

} // namespace popart
