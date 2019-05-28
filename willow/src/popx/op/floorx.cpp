#include <popops/ElementWise.hpp>
#include <poponnx/op/floor.hpp>
#include <poponnx/popx/devicex.hpp>
#include <poponnx/popx/op/floorx.hpp>
#include <poponnx/popx/opxmanager.hpp>

namespace poponnx {
namespace popx {

poplar::Tensor FloorComputex::outplace(poplar::program::Sequence &prog,
                                       poplar::Graph &graph,
                                       const poplar::Tensor &tensor,
                                       const std::string &s) const {

  return popops::map(graph, popops::expr::UnaryOpType::FLOOR, tensor, prog, s);
}

void FloorComputex::inplace(poplar::program::Sequence &prog,
                            poplar::Graph &graph,
                            const poplar::Tensor &tensor,
                            const std::string &s) const {

  popops::mapInPlace(graph, popops::expr::UnaryOpType::FLOOR, tensor, prog, s);
}

FloorOpx::FloorOpx(Op *op, Devicex *devicex)
    : ElementWiseUnaryOutplaceOpx(op, devicex, FloorComputex::get()) {
  verifyOp<FloorOp>(op, {Onnx::Operators::Floor_1, Onnx::Operators::Floor_6});
}

FloorInplaceOpx::FloorInplaceOpx(Op *op, Devicex *devicex)
    : ElementWiseUnaryInplaceOpx(op, devicex, FloorComputex::get()) {
  verifyOp<FloorInplaceOp>(op, Onnx::CustomOperators::FloorInplace);
}

namespace {
OpxCreator<FloorOpx> FloorOpxCreator({Onnx::Operators::Floor_1,
                                      Onnx::Operators::Floor_6});
OpxCreator<FloorInplaceOpx>
    floorxInplaceOpxCreator(Onnx::CustomOperators::FloorInplace);
} // namespace

} // namespace popx
} // namespace poponnx
