// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <popart/op/elementwise.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/irlowering.hpp>
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
      op_p->outInfo(ElementWiseBinaryBaseOp::getOutIndex())) {
    return InputCreatorType::Deadend;
  }

  const auto &settings = this->op_p->settings;
  const auto arg0Idx   = ElementWiseBinaryBaseOp::getArg0InIndex();
  const auto arg1Idx   = ElementWiseBinaryBaseOp::getArg1InIndex();

  const auto itArg0 = settings.inferTensorMappingToFrom.find(arg0Idx);
  const auto itArg1 = settings.inferTensorMappingToFrom.find(arg1Idx);

  const bool inferArg0FromArg1 =
      itArg0 != settings.inferTensorMappingToFrom.end() &&
      itArg0->second == arg1Idx;
  const bool inferArg1FromArg0 =
      itArg1 != settings.inferTensorMappingToFrom.end() &&
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

  const auto &settings = this->op_p->settings;
  const auto arg0Idx   = ElementWiseBinaryBaseOp::getArg0InIndex();
  const auto arg1Idx   = ElementWiseBinaryBaseOp::getArg1InIndex();

  std::vector<TensorId> mustExist;

  auto it = settings.inferTensorMappingToFrom.find(index);

  if (it != settings.inferTensorMappingToFrom.end() &&
      ((it->first == arg0Idx && it->second == arg1Idx) ||
       (it->first == arg1Idx && it->second == arg0Idx))) {
    mustExist.push_back(op_p->input->tensor(it->second)->id);
  }

  return mustExist;
}

poplar::Tensor
ElementWiseBinaryOpx::createInput(InIndex index,
                                  const std::string &name) const {

  const auto arg0Idx = ElementWiseBinaryBaseOp::getArg0InIndex();
  const auto arg1Idx = ElementWiseBinaryBaseOp::getArg1InIndex();

  if (index == arg0Idx) {
    if (dv_p->lowering().tensors().contains(op_p->input->id(arg1Idx))) {
      return graph().clone(getInTensor(arg1Idx), name);
    }
  }

  if (index == arg1Idx) {
    if (dv_p->lowering().tensors().contains(op_p->input->id(arg0Idx))) {
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
    outTensor = cx->outplace(prog,
                             graph(),
                             outTensor,
                             getDebugNameAndId(),
                             "nonLinearityOutplaceFallback");
  } else {
    cx->inplace(
        prog, graph(), outTensor, getDebugNameAndId(), "nonLinearityInplace");
  }
  outTensor = cx->reshape(outTensor);
  if (hasInViewChangers(ElementWiseUnaryOp::getInIndex())) {
    setOutViewChangers(ElementWiseUnaryOp::getOutIndex(),
                       getInViewChangers(ElementWiseUnaryOp::getInIndex()));
  }
  setOutTensor(ElementWiseUnaryOp::getOutIndex(), outTensor);
}

void ElementWiseUnaryOutplaceOpx::grow(poplar::program::Sequence &prog) const {
  auto outTensor = cx->outplace(prog,
                                graph(),
                                getInTensor(ElementWiseUnaryOp::getInIndex()),
                                getDebugNameAndId(),
                                "nonLinearityOutplace");

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

poplar::Tensor
EwuComputex::cloneNcopy(poplar::program::Sequence &prog,
                        poplar::Graph &graph,
                        const poplar::Tensor &tensor,
                        const poplar::DebugNameAndId &dnai) const {
  auto outTensor = graph.clone(tensor, {dnai});
  poplar::program::Copy copyProg(tensor, outTensor, false, {dnai});
  prog.add(copyProg);
  return outTensor;
}

poplar::Tensor EwuComputex::outplace(poplar::program::Sequence &prog,
                                     poplar::Graph &graph,
                                     const poplar::Tensor &tensor,
                                     const poplar::DebugNameAndId &dnai,
                                     const std::string &debug_prefix) const {
  auto out_tensor = cloneNcopy(prog, graph, tensor, dnai);
  inplace(prog, graph, out_tensor, dnai, debug_prefix);
  return out_tensor;
}

poplar::Tensor EwuComputex::coerceTo2D(const poplar::Tensor &t, int64_t axis) {
  const auto in_shape = t.shape();
  auto k              = in_shape.begin();
  std::advance(k, axis);

  auto n = std::accumulate(
      in_shape.begin(), k, std::size_t{1}, std::multiplies<std::size_t>());
  auto d = std::accumulate(
      k, in_shape.end(), std::size_t{1}, std::multiplies<std::size_t>());
  return t.reshape({n, d});
}

bool EwbComputex::inplaceSupported() const {
  return inplacePolicy != EwbComputex::InplacePolicy::NEVER;
}

InIndex EwbComputex::getInplaceArgInIndex() const {
  if (inplacePolicy == InplacePolicy::LHS) {
    return ElementWiseBinaryBaseOp::getArg0InIndex();
  } else if (inplacePolicy == InplacePolicy::RHS) {
    return ElementWiseBinaryBaseOp::getArg1InIndex();
  } else {
    throw internal_error(
        "Invalid InplacePolicy. This class instance was not configured for "
        "inplacing and is attempting to compute in-place");
  }
}

InIndex EwbComputex::getOutplaceArgInIndex() const {
  // The out-of-place index is the input that isn't in-place
  const auto inplaceIdx = getInplaceArgInIndex();
  const auto arg0Idx    = ElementWiseBinaryBaseOp::getArg0InIndex();
  const auto arg1Idx    = ElementWiseBinaryBaseOp::getArg1InIndex();
  return inplaceIdx == arg0Idx ? arg1Idx : arg0Idx;
}

void ElementWiseBinaryOutplaceOpx::grow(poplar::program::Sequence &prog) const {
  if (cx->inplaceSupported()) {
    throw internal_error(
        "Operation {} was configured for inplacing and attempting "
        "to compute out-of-place",
        debugContext().getPathName());
  }

  const auto arg0Idx = ElementWiseBinaryBaseOp::getArg0InIndex();
  const auto arg1Idx = ElementWiseBinaryBaseOp::getArg1InIndex();
  const auto outIdx  = ElementWiseBinaryBaseOp::getOutIndex();

  auto outTensor = cx->outplace(prog,
                                graph(),
                                getInTensor(arg0Idx),
                                getInTensor(arg1Idx),
                                getDebugNameAndId(),
                                "");

  if (hasInViewChangers(ElementWiseBinaryOp::getArg0InIndex())) {
    setOutViewChangers(
        ElementWiseBinaryOp::getOutIndex(),
        getInViewChangers(ElementWiseBinaryOp::getArg0InIndex()));
  }
  setOutTensor(outIdx, outTensor);
}

void ElementWiseBinaryInplaceOpx::grow(poplar::program::Sequence &prog) const {
  if (!cx->inplaceSupported()) {
    throw error("Invalid operation {} was not configured for inplacing and "
                "attempting to compute in-place",
                debugContext().getPathName());
  }

  constexpr unsigned maxTileImbalance = 150000;
  bool canComputeInplace              = true;

  auto tInOut    = getInTensor(cx->getInplaceArgInIndex());
  const auto tIn = getInTensor(cx->getOutplaceArgInIndex());
  auto &g        = graph();

  if (!tInOut.isParallelWriteable()) {
    logging::debug(
        "Unable to inplace operation {}, tensor is not parallel writeable",
        debugContext().getPathName());
    canComputeInplace = false;
  } else if (poputil::getTileImbalance(g, tInOut) > maxTileImbalance) {
    logging::debug("Unable to inplace operation {}, tensor tile imbalance ({}) "
                   "is too high",
                   debugContext().getPathName(),
                   poputil::getTileImbalance(g, tInOut));
    canComputeInplace = false;
  }

  const auto outIdx = ElementWiseBinaryBaseOp::getOutIndex();

  if (canComputeInplace) {
    cx->inplace(prog, g, tInOut, tIn, getDebugNameAndId(), "");
    tInOut = tInOut.reshape(outInfo(outIdx).shape_szt());
  } else {
    tInOut = cx->outplace(prog, g, tInOut, tIn, getDebugNameAndId(), "");
  }

  if (hasInViewChangers(ElementWiseBinaryOp::getArg0InIndex())) {
    setOutViewChangers(
        ElementWiseBinaryOp::getOutIndex(),
        getInViewChangers(ElementWiseBinaryOp::getArg0InIndex()));
  }
  setOutTensor(outIdx, tInOut);
}

BinaryComparisonOpx::BinaryComparisonOpx(Op *op, Devicex *devicex)
    : Opx(op, devicex) {}

} // namespace popx
} // namespace popart
