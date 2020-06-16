// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <popart/op/elementwise.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/elementwisex.hpp>

namespace popart {
namespace popx {

ElementWiseUnaryOutplaceOpx::ElementWiseUnaryOutplaceOpx(
    Op *op,
    Devicex *devx,
    std::unique_ptr<EwuComputex> cx_)
    : ElementWiseUnaryOpx(op, devx), cx(std::move(cx_)) {}

ElementWiseUnaryOpx::ElementWiseUnaryOpx(Op *op, Devicex *devicex)
    : Opx(op, devicex) {}

InputCreatorType ElementWiseUnaryOpx::getInputCreatorType(InIndex) const {
  return InputCreatorType::CanUnwind;
}

poplar::Tensor ElementWiseUnaryOpx::unwindTensorLayout(poplar::Tensor tensor,
                                                       InIndex,
                                                       OutIndex) const {
  return tensor;
}

view::RegMap ElementWiseUnaryOpx::unwindRegion(InIndex, OutIndex) const {
  return [](const view::Region &r) { return view::Regions(1, r); };
}

ElementWiseBinaryOpx::ElementWiseBinaryOpx(Op *op, Devicex *devicex)
    : Opx(op, devicex) {}

InputCreatorType
ElementWiseBinaryOpx::getInputCreatorType(InIndex index) const {
  // Check shape doesn't change due to numpy-style broadcasting.
  // Design choice: even without broadcasting, it is possible for the
  // two inputs (of same shape) have different layout.
  // The poplar binary op can choose the layout of the output to take
  // the layout of either input.
  // However, let's layout both inputs in the same way. That way we can
  // definitely unwind through this opx, and it will also be efficient
  // when performing the op.
  if (op_p->inInfo(index) !=
      op_p->outInfo(ElementWiseBinaryOp::getOutIndex())) {
    return InputCreatorType::Deadend;
  }

  ElementWiseBinaryOp *op = dynamic_cast<ElementWiseBinaryOp *>(this->op_p);
  const auto arg0Idx      = op->getArg0InIndex();
  const auto arg1Idx      = op->getArg1InIndex();

  const auto itArg0 = op->settings.inferTensorMappingToFrom.find(arg0Idx);
  const auto itArg1 = op->settings.inferTensorMappingToFrom.find(arg1Idx);

  const bool inferArg0FromArg1 =
      itArg0 != op->settings.inferTensorMappingToFrom.end() &&
      itArg0->second == arg1Idx;
  const bool inferArg1FromArg0 =
      itArg1 != op->settings.inferTensorMappingToFrom.end() &&
      itArg1->second == arg0Idx;

  if (index == arg0Idx) {
    if (inferArg0FromArg1) {
      return InputCreatorType::CanCreateOrUnwind;
    } else if (inferArg1FromArg0) {
      return InputCreatorType::Deadend;
    }
  } else if (index == arg1Idx) {
    if (inferArg1FromArg0) {
      return InputCreatorType::CanCreateOrUnwind;
    } else if (inferArg0FromArg1) {
      return InputCreatorType::Deadend;
    }
  }

  return InputCreatorType::CanUnwind;
}

std::vector<TensorId>
ElementWiseBinaryOpx::mustExistBeforeCreate(InIndex index) const {

  ElementWiseBinaryOp *op = dynamic_cast<ElementWiseBinaryOp *>(this->op_p);
  const auto arg0Idx      = op->getArg0InIndex();
  const auto arg1Idx      = op->getArg1InIndex();

  std::vector<TensorId> mustExist;

  auto it = op->settings.inferTensorMappingToFrom.find(index);

  if (it != op->settings.inferTensorMappingToFrom.end() &&
      ((it->first == arg0Idx && it->second == arg1Idx) ||
       (it->first == arg1Idx && it->second == arg0Idx))) {
    mustExist.push_back(op->input->tensor(it->second)->id);
  }

  return mustExist;
}

poplar::Tensor
ElementWiseBinaryOpx::createInput(InIndex index,
                                  const std::string &name) const {

  ElementWiseBinaryOp *op = dynamic_cast<ElementWiseBinaryOp *>(this->op_p);
  const auto arg0Idx      = op->getArg0InIndex();
  const auto arg1Idx      = op->getArg1InIndex();

  if (index == arg0Idx) {
    if (dv_p->tensors.contains(op_p->input->id(arg1Idx))) {
      return graph().clone(getInTensor(arg1Idx), name);
    }
  }

  if (index == arg1Idx) {
    if (dv_p->tensors.contains(op_p->input->id(arg0Idx))) {
      return graph().clone(getInTensor(arg0Idx), name);
    }
  }

  throw error("ElementWiseBinaryOpx::createInput : Invalid index = " +
              std::to_string(index));
}

void ElementWiseUnaryInplaceOpx::grow(poplar::program::Sequence &prog) const {
  auto outTensor = getInTensor(ElementWiseUnaryOp::getInIndex());

  // if all of the elements in the tensor are distinct in memory,
  // them we can use the poplar inplace version. Otherwise, we must
  // use a non-inplace version.  See T7110 for a possible improvement
  if (!outTensor.isParallelWriteable()) {
    outTensor = cx->outplace(
        prog, graph(), outTensor, debugPrefix("nonLinearityInplace"));
  } else {
    cx->inplace(
        prog, graph(), outTensor, debugPrefix("nonLinearityOutplaceFallback"));
  }
  outTensor = cx->reshape(outTensor);
  setOutTensor(ElementWiseUnaryOp::getOutIndex(), outTensor);
}

void ElementWiseUnaryOutplaceOpx::grow(poplar::program::Sequence &prog) const {
  auto outTensor = cx->outplace(prog,
                                graph(),
                                getInTensor(ElementWiseUnaryOp::getInIndex()),
                                debugPrefix("nonLinearityOutplace"));

  outTensor = cx->reshape(outTensor);
  setOutTensor(ElementWiseUnaryOp::getOutIndex(), outTensor);
}

poplar::Tensor ElementWiseBinaryOpx::unwindTensorLayout(poplar::Tensor tensor,
                                                        InIndex,
                                                        OutIndex) const {
  return tensor;
}

view::RegMap ElementWiseBinaryOpx::unwindRegion(InIndex, OutIndex) const {
  return [](const view::Region &r) { return view::Regions(1, r); };
}

poplar::Tensor EwuComputex::cloneNcopy(poplar::program::Sequence &prog,
                                       poplar::Graph &graph,
                                       const poplar::Tensor &tensor) const {

  auto outTensor = graph.clone(tensor);
  poplar::program::Copy copyProg(tensor, outTensor);
  prog.add(copyProg);
  return outTensor;
}

poplar::Tensor EwuComputex::outplace(poplar::program::Sequence &prog,
                                     poplar::Graph &graph,
                                     const poplar::Tensor &tensor,
                                     const std::string &debug_prefix) const {
  auto out_tensor = cloneNcopy(prog, graph, tensor);
  inplace(prog, graph, out_tensor, debug_prefix);
  return out_tensor;
}

BinaryComparisonOpx::BinaryComparisonOpx(Op *op, Devicex *devicex)
    : Opx(op, devicex) {}

} // namespace popx
} // namespace popart
