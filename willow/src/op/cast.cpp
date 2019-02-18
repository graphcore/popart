#include <poponnx/makeunique.hpp>
#include <poponnx/onnxutil.hpp>
#include <poponnx/op/cast.hpp>
#include <poponnx/opmanager.hpp>

namespace poponnx {

CastOp::CastOp(const OperatorIdentifier &_opid,
               DataType _to,
               const Op::Settings &settings_)
    : Op(_opid, settings_), to(_to) {}

std::unique_ptr<Op> CastOp::clone() const { return make_unique<CastOp>(*this); }

std::vector<std::unique_ptr<Op>> CastOp::getGradOps() {
  throw error("CastOp should be removed by const folding");
}

void CastOp::setup() { outInfo(getOutIndex()) = {to, inShape(getInIndex())}; }

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

      return make_unique<CastOp>(opid, dt_to, settings);
    },
    true);
} // namespace

} // namespace poponnx
