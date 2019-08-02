#include <popart/error.hpp>
#include <popart/op/not.hpp>
#include <popart/popx/devicex.hpp>

#include <popart/popx/op/notx.hpp>
#include <popart/popx/opxmanager.hpp>

#include <popops/ElementWise.hpp>

namespace popart {
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
                     debugPrefix()));
}

namespace {

OpxCreator<NotOpx> greaterOpxCreator_1(Onnx::Operators::Not_1);

} // namespace

} // namespace popx
} // namespace popart
