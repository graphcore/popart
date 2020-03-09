#include <popart/error.hpp>
#include <popart/op/dynamic/dynamicslice.hpp>

#include <popart/popx/devicex.hpp>
#include <popart/popx/op/dynamic/dynamicslicex.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorindex.hpp>

#include <popops/DynamicSlice.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Zero.hpp>

namespace popart {
namespace popx {

DynamicSliceOpx::DynamicSliceOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<DynamicSliceBaseOp>(op);
  inputCreatorPriority = -1.0;
}

void DynamicSliceOpx::grow(poplar::program::Sequence &prog) const {
  auto &op    = getOp<DynamicSliceBaseOp>();
  auto tensor = getInTensor(DynamicSliceBaseOp::getInIndex());
  auto index  = getInTensor(DynamicSliceBaseOp::getIndexInIndex());

  std::vector<size_t> paxes(op.getAxes().begin(), op.getAxes().end());
  std::vector<size_t> psizes(op.getSizes().begin(), op.getSizes().end());

  auto s = popops::dynamicSlice(
      graph(),
      tensor,
      index,
      paxes,
      psizes,
      prog,
      debugPrefix("dynamic_slice_" +
                  op.inId(DynamicSliceBaseOp::getInIndex())));

  setOutTensor(DynamicSliceBaseOp::getOutIndex(), s);
}

InputCreatorType DynamicSliceOpx::getInputCreatorType(InIndex index) const {
  return index == DynamicSliceBaseOp::getInIndex()
             ? InputCreatorType::CANCREATE_OR_UNWIND
             : Opx::getInputCreatorType(index);
}

poplar::Tensor DynamicSliceOpx::createInput(InIndex index,
                                            const std::string &name) const {
  auto &op = getOp<DynamicSliceBaseOp>();

  if (index == DynamicSliceBaseOp::getInIndex()) {
    auto outInfo     = op.outInfo(DynamicSliceBaseOp::getOutIndex());
    auto sliceTensor = graph().addVariable(
        popType(outInfo),
        outInfo.shape_szt(),
        op.inId(DynamicSliceBaseOp::getInIndex()) + "_slice");
    dv_p->getLinearMapper().mapTensor(graph(), sliceTensor);
    auto inShape = op.inShape(DynamicSliceBaseOp::getInIndex());

    std::vector<size_t> paxes(op.getAxes().begin(), op.getAxes().end());
    std::vector<size_t> psizes(op.getSizes().begin(), op.getSizes().end());

    std::vector<size_t> numSlices(paxes.size(), 0);
    for (size_t i = 0; i < paxes.size(); ++i) {
      numSlices[i] = inShape[paxes[i]] / psizes[i];
    }

    return popops::createSliceableTensorFromSlice(
        graph(), sliceTensor, paxes, numSlices, name);
  }

  throw error("DynamicSliceOpx::createInput : Invalid index = " +
              std::to_string(index));
}

poplar::Tensor DynamicSliceOpx::unwindTensorLayout(poplar::Tensor tensor,
                                                   InIndex,
                                                   OutIndex) const {
  auto &op      = getOp<DynamicSliceBaseOp>();
  auto outShape = tensor.shape();
  auto inShape  = op.inShape(DynamicSliceBaseOp::getInIndex());

  std::vector<size_t> paxes(op.getAxes().begin(), op.getAxes().end());
  std::vector<size_t> psizes(op.getSizes().begin(), op.getSizes().end());

  std::vector<size_t> numSlices(paxes.size(), 0);
  for (size_t i = 0; i < paxes.size(); ++i) {
    numSlices[i] = inShape[paxes[i]] / psizes[i];
  }
  return popops::createSliceableTensorFromSlice(
      graph(), tensor, paxes, numSlices);
}

view::RegMap DynamicSliceOpx::unwindRegion(InIndex index, OutIndex) const {
  DynamicSliceBaseOp *op = dynamic_cast<DynamicSliceBaseOp *>(this->op_p);
  auto shape             = op->inShape(index);
  return [shape](const view::Region &r) {
    return view::Regions(1, view::Region::getFull(shape));
  };
}

namespace {
// Ops
OpxCreator<DynamicSliceOpx>
    dynamicSliceOpxCreator(Onnx::CustomOperators::DynamicSlice_1);
} // namespace

} // namespace popx
} // namespace popart
