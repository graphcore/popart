#include <poponnx/op/split.hpp>
#include <poponnx/popx/op/splitx.hpp>
#include <poponnx/popx/opxmanager.hpp>
#include <poponnx/tensorindex.hpp>

namespace poponnx {

namespace popx {

SplitOpx::SplitOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<SplitOp>(op, Onnx::Operators::Split_2);
}

void SplitOpx::grow(poplar::program::Sequence &prog) const {
  auto &splitOp     = getOp<SplitOp>();
  auto input        = getInTensor(SplitOp::getInIndex());
  unsigned int axis = static_cast<unsigned int>(splitOp.getAxis());

  unsigned int start = 0;
  auto splitSizes    = splitOp.getSplitSizes();
  for (int i = 0; i < splitSizes.size(); i++) {
    unsigned int end = start + static_cast<unsigned int>(splitSizes.at(i));
    auto t           = input.slice(start, end, axis);

    setOutTensor(i, cloneNcopy(prog, t));

    start = end;
  }
}

namespace {
OpxCreator<SplitOpx> splitOpxCreator(Onnx::Operators::Split_2);
} // namespace

} // namespace popx
} // namespace poponnx
