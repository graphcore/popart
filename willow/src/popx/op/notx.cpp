#include <poponnx/error.hpp>
#include <poponnx/op/not.hpp>
#include <poponnx/popx/devicex.hpp>

#include <poponnx/popx/op/notx.hpp>
#include <poponnx/popx/opxmanager.hpp>

#include <popops/ElementWise.hpp>

namespace poponnx {
namespace popx {

NotOpx::NotOpx(Op *op, Devicex *devicex) : ElementWiseUnaryOpx(op, devicex) {
  verifyOp<NotOp>(op, {Onnx::Operators::Not_1});
}

void NotOpx::grow(poplar::program::Sequence &prog) const {

  insert(outId(NotOp::getOutIndex()),
         popops::map(graph(),
                     popops::expr::UnaryOpType::LOGICAL_NOT,
                     get(inId(NotOp::getInIndex())),
                     prog,
                     idStr()));
}

namespace {

OpxCreator<NotOpx> greaterOpxCreator_1(Onnx::Operators::Not_1);

} // namespace

} // namespace popx
} // namespace poponnx
