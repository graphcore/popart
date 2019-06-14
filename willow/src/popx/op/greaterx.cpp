#include <popops/Cast.hpp>
#include <poponnx/error.hpp>
#include <poponnx/op/greater.hpp>
#include <poponnx/popx/devicex.hpp>

#include <poponnx/popx/op/greaterx.hpp>
#include <poponnx/popx/opxmanager.hpp>

#include <popops/ElementWise.hpp>

namespace poponnx {
namespace popx {

GreaterOpx::GreaterOpx(Op *op, Devicex *devicex)
    : BinaryComparisonOpx(op, devicex) {
  verifyOp<GreaterOp>(op,
                      {Onnx::Operators::Greater_7, Onnx::Operators::Greater_9});
}

void GreaterOpx::grow(poplar::program::Sequence &prog) const {

  insert(outId(GreaterOp::getOutIndex()),
         popops::map(graph(),
                     popops::expr::BinaryOpType::GREATER_THAN,
                     get(inId(GreaterOp::getArg0InIndex())),
                     get(inId(GreaterOp::getArg1InIndex())),
                     prog,
                     idStr()));
}

namespace {

OpxCreator<GreaterOpx> greaterOpxCreator_7(Onnx::Operators::Greater_7);
OpxCreator<GreaterOpx> greaterOpxCreator_9(Onnx::Operators::Greater_9);

} // namespace

} // namespace popx
} // namespace poponnx
