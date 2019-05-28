#include <popops/ElementWise.hpp>
#include <poponnx/op/ceil.hpp>
#include <poponnx/popx/devicex.hpp>
#include <poponnx/popx/op/ceilx.hpp>
#include <poponnx/popx/opxmanager.hpp>

namespace poponnx {
namespace popx {

poplar::Tensor CeilComputex::outplace(poplar::program::Sequence &prog,
                                      poplar::Graph &graph,
                                      const poplar::Tensor &tensor,
                                      const std::string &s) const {

  return popops::map(graph, popops::expr::UnaryOpType::CEIL, tensor, prog, s);
}

void CeilComputex::inplace(poplar::program::Sequence &prog,
                           poplar::Graph &graph,
                           const poplar::Tensor &tensor,
                           const std::string &s) const {

  popops::mapInPlace(graph, popops::expr::UnaryOpType::CEIL, tensor, prog, s);
}

CeilOpx::CeilOpx(Op *op, Devicex *devicex)
    : ElementWiseUnaryOutplaceOpx(op, devicex, CeilComputex::get()) {
  verifyOp<CeilOp>(op, {Onnx::Operators::Ceil_1, Onnx::Operators::Ceil_6});
}

CeilInplaceOpx::CeilInplaceOpx(Op *op, Devicex *devicex)
    : ElementWiseUnaryInplaceOpx(op, devicex, CeilComputex::get()) {
  verifyOp<CeilInplaceOp>(op, Onnx::CustomOperators::CeilInplace);
}

namespace {
OpxCreator<CeilOpx> ceilOpxCreator({Onnx::Operators::Ceil_1,
                                    Onnx::Operators::Ceil_6});
OpxCreator<CeilInplaceOpx>
    ceilxInplaceOpxCreator(Onnx::CustomOperators::CeilInplace);
} // namespace

} // namespace popx
} // namespace poponnx
