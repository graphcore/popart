// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <popart/error.hpp>
#include <popart/ir.hpp>
#include <popart/op/add.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/addx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorindex.hpp>

#include <popops/ElementWise.hpp>
#include <poputil/TileMapping.hpp>

namespace popart {
namespace popx {

const unsigned max_tile_imbalance = 150000;

static poplar::Tensor addInplace(poplar::Graph &graph,
                                 const poplar::Tensor &t_inout,
                                 const poplar::Tensor &t_in,
                                 poplar::program::Sequence &prog,
                                 const std::string debug_id) {
  bool can_do_inplace = true;
  if (!t_inout.isParallelWriteable()) {
    logging::debug(
        "Unable to inplace add operation {}, tensor is not parallel writeable",
        debug_id);
    can_do_inplace = false;
  } else if (poputil::getTileImbalance(graph, t_inout) > max_tile_imbalance) {
    logging::debug("Unable to inplace add operation {}, tensor tile imbalance "
                   "({}) is too high",
                   debug_id,
                   poputil::getTileImbalance(graph, t_inout));
    can_do_inplace = false;
  }

  if (can_do_inplace) {
    popops::mapInPlace(
        graph, popops::expr::BinaryOpType::ADD, t_inout, t_in, prog, debug_id);
    return t_inout;
  } else {
    return popops::map(
        graph, popops::expr::BinaryOpType::ADD, t_inout, t_in, prog, debug_id);
  }
}

AddOpx::AddOpx(Op *op, Devicex *devicex) : ElementWiseBinaryOpx(op, devicex) {
  verifyOp<AddOp>(op,
                  {Onnx::Operators::Add_6,
                   Onnx::Operators::Add_7,
                   Onnx::CustomOperators::AddLhsInplace,
                   Onnx::CustomOperators::AddRhsInplace});
}

void AddOpx::grow(poplar::program::Sequence &prog) const {

  setOutTensor(AddOp::getOutIndex(),
               popops::map(graph(),
                           popops::expr::BinaryOpType::ADD,
                           getInTensor(AddOp::getArg0InIndex()),
                           getInTensor(AddOp::getArg1InIndex()),
                           prog,
                           debugPrefix()));
}

InputCreatorType AddOpx::getInputCreatorType(InIndex index) const {
  AddOp *op = dynamic_cast<AddOp *>(this->op_p);

  // TODO: T17972 Allowing add (in particular lhs, rhs inplace adds) leads to
  // inefficient sub graph copying. Investigate why, then remove the below logic
  // once fixed.
  if (!(op_p->getIr().getSessionOptions().decomposeGradSum ||
        op_p->getIr().getSessionOptions().batchSerializationFactor > 0)) {
    return InputCreatorType::Deadend;
  }

  // Check shape doesn't change due to numpy-style broadcasting.
  if (op_p->inInfo(index) != op_p->outInfo(AddOp::getOutIndex())) {
    return InputCreatorType::Deadend;
  }

  auto itArg0 =
      op->settings.inferTensorMappingToFrom.find(AddOp::getArg0InIndex());
  auto itArg1 =
      op->settings.inferTensorMappingToFrom.find(AddOp::getArg1InIndex());

  bool inferArg0FromArg1 =
      itArg0 != op->settings.inferTensorMappingToFrom.end() &&
      itArg0->second == AddOp::getArg1InIndex();
  bool inferArg1FromArg0 =
      itArg1 != op->settings.inferTensorMappingToFrom.end() &&
      itArg1->second == AddOp::getArg0InIndex();

  if (index == AddOp::getArg0InIndex()) {
    if (inferArg0FromArg1) {
      return InputCreatorType::CanCreateOrUnwind;
    } else if (inferArg1FromArg0) {
      return InputCreatorType::Deadend;
    } else {
      return InputCreatorType::CanUnwind;
    }
  }

  if (index == AddOp::getArg1InIndex()) {
    if (inferArg1FromArg0) {
      return InputCreatorType::CanCreateOrUnwind;
    } else if (inferArg0FromArg1) {
      return InputCreatorType::Deadend;
    } else {
      return InputCreatorType::CanUnwind;
    }
  }

  return Opx::getInputCreatorType(index);
}

poplar::Tensor AddOpx::createInput(InIndex index,
                                   const std::string &name) const {

  if (index == AddOp::getArg0InIndex()) {
    if (dv_p->tensors.contains(op_p->input->id(AddOp::getArg1InIndex()))) {
      return graph().clone(getInTensor(AddOp::getArg1InIndex()), name);
    }
  }

  if (index == AddOp::getArg1InIndex()) {
    if (dv_p->tensors.contains(op_p->input->id(AddOp::getArg0InIndex()))) {
      return graph().clone(getInTensor(AddOp::getArg0InIndex()), name);
    }
  }

  throw error("AddOpx::createInput : Invalid index = " + std::to_string(index));
}

std::vector<TensorId> AddOpx::mustExistBeforeCreate(InIndex index) const {
  AddOp *op = dynamic_cast<AddOp *>(this->op_p);

  std::vector<TensorId> mustExist;

  auto it = op->settings.inferTensorMappingToFrom.find(index);

  if (it != op->settings.inferTensorMappingToFrom.end() &&
      ((it->first == AddOp::getArg0InIndex() &&
        it->second == AddOp::getArg1InIndex()) ||
       (it->first == AddOp::getArg1InIndex() &&
        it->second == AddOp::getArg0InIndex()))) {
    mustExist.push_back(op->input->tensor(it->second)->id);
  }

  return mustExist;
}

AddLhsInplaceOpx::AddLhsInplaceOpx(Op *op, Devicex *devicex)
    : AddOpx(op, devicex) {
  verifyOp<AddLhsInplaceOp>(op);
}

void AddLhsInplaceOpx::grow(poplar::program::Sequence &prog) const {
  auto out = addInplace(graph(),
                        getInTensor(AddLhsInplaceOp::getArg0InIndex()),
                        getInTensor(AddLhsInplaceOp::getArg1InIndex()),
                        prog,
                        debugPrefix());

  out = out.reshape(outInfo(AddLhsInplaceOp::getOutIndex()).shape_szt());
  setOutTensor(AddLhsInplaceOp::getOutIndex(), out);
}

AddRhsInplaceOpx::AddRhsInplaceOpx(Op *op, Devicex *devicex)
    : AddOpx(op, devicex) {
  verifyOp<AddRhsInplaceOp>(op);
}

void AddRhsInplaceOpx::grow(poplar::program::Sequence &prog) const {
  auto out = addInplace(graph(),
                        getInTensor(AddRhsInplaceOp::getArg1InIndex()),
                        getInTensor(AddRhsInplaceOp::getArg0InIndex()),
                        prog,
                        debugPrefix());

  out = out.reshape(outInfo(AddRhsInplaceOp::getOutIndex()).shape_szt());
  setOutTensor(AddRhsInplaceOp::getOutIndex(), out);
}

AddArg0GradOpx::AddArg0GradOpx(Op *op, Devicex *devicex)
    : ReduceSumOpx(op, devicex) {
  verifyOp<AddArg0GradOp>(op, Onnx::GradOperators::AddArg0Grad);
}

AddArg1GradOpx::AddArg1GradOpx(Op *op, Devicex *devicex)
    : ReduceSumOpx(op, devicex) {
  verifyOp<AddArg1GradOp>(op, Onnx::GradOperators::AddArg1Grad);
}

namespace {
// OpxCreator<AddOpx> addOpxCreator({Onnx::Operators::Add_6,
// Onnx::Operators::Add_7});

OpxCreator<AddOpx> addOpxCreator_6(Onnx::Operators::Add_6);
OpxCreator<AddOpx> addOpxCreator_7(Onnx::Operators::Add_7);
OpxCreator<AddLhsInplaceOpx>
    addLhsInplaceOpxCreator(Onnx::CustomOperators::AddLhsInplace);
OpxCreator<AddRhsInplaceOpx>
    addRhsInplaceOpxCreator(Onnx::CustomOperators::AddRhsInplace);

OpxCreator<AddArg0GradOpx>
    addArg0GradOpxCreator(Onnx::GradOperators::AddArg0Grad);
OpxCreator<AddArg1GradOpx>
    addArg1GradOpxCreator(Onnx::GradOperators::AddArg1Grad);
} // namespace

} // namespace popx
} // namespace popart
