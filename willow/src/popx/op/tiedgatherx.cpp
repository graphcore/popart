// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <ext/new_allocator.h>
#include <tuple>
#include <vector>
#include <poplar/Graph.hpp>
#include <poplar/Tensor.hpp>
#include <poplar/Type.hpp>
#include <poplin/MatMul.hpp>
#include <popops/Gather.hpp>
#include <popart/op/tiedgather.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/irlowering.hpp>
#include <popart/popx/op/tiedgatherx.hpp>
#include <popart/popx/opxmanager.hpp>

#include "popart/error.hpp"
#include "popart/graphcoreoperators.hpp"
#include "popart/logging.hpp"
#include "popart/names.hpp"
#include "popart/popx/debugcontextx.hpp"
#include "popart/popx/op/gatherx.hpp"
#include "popart/popx/opx.hpp"
#include "popart/tensorinfo.hpp"
#include "popart/util.hpp"

namespace poplar {
namespace program {
class Sequence;
} // namespace program
} // namespace poplar

namespace popart {
class Op;

namespace popx {

TiedGatherOpx::TiedGatherOpx(Op *op, Devicex *device)
    : GatherBaseOpx(op, device) {
  verifyOp<TiedGatherOp>(op, {Onnx::CustomOperators::TiedGather});

  setCommonMembersPostVerify(op);

  /*
    PyTorch BERT was developed based on a bug in DetachOp that caused the tile
    layout of the weight to be determined by the MatMul, not the TiedGather.
    This costs extra cycles due to the exchange, but actually has lower
    memory footprint.

    In order to natively support PyTorch BERT in the short-term, we thus have to
    disable the TiedGather input creator, so we get the same behaviour that
    resulted from the DetachOp bug.
   */
  inputCreatorPriority = -1;
}

InputCreatorType TiedGatherOpx::getInputCreatorType(int index0) const {
  return index0 == TiedGatherOp::dataInIndex()
             ? InputCreatorType::CanCreate
             : Opx::getInputCreatorType(index0);
}

poplar::Tensor
TiedGatherOpx::createInput(const InIndex index,
                           const poplar::DebugNameAndId &dnai) const {
  logging::devicex::debug(
      "TiedGather asked to create index {}: name {}", index, dnai);

  if (index != TiedGatherOp::dataInIndex()) {
    throw error("TiedGatherOpx::createInput: Cannot create input {}", index);
  }

  auto inputInfo  = inInfo(TiedGatherOp::indicesInIndex());
  auto weightInfo = inInfo(TiedGatherOp::dataInIndex());

  unsigned inputSize   = inputInfo.nelms();
  unsigned inChannels  = weightInfo.dim(getOp<TiedGatherOp>().getAxis());
  unsigned outChannels = weightInfo.nelms() / inChannels;

  std::vector<std::size_t> lhsShape = {inputSize, inChannels};
  std::vector<std::size_t> rhsShape = {inChannels, outChannels};

  return poplin::createMatMulInputRHS(graph(),
                                      popType(weightInfo),
                                      lhsShape,
                                      rhsShape,
                                      dnai,
                                      {},
                                      &dv_p->matmulCache);
}

void TiedGatherOpx::grow(poplar::program::Sequence &prog) const {
  const auto indicesShape = inShape(TiedGatherOp::indicesInIndex());
  const auto outputShape =
      vXtoY<int64_t, std::size_t>(outShape(TiedGatherOp::outIndex()));

  auto op       = getOp<TiedGatherOp>();
  unsigned axis = op.getAxis();
  auto indices  = getInTensor(TiedGatherOp::indicesInIndex());
  auto data     = getInTensor(TiedGatherOp::dataInIndex());

  // If there are no indices, return an empty tensor of the appropriate
  // shape
  if (indices.numElements() == 0) {
    auto result = graph().addVariable(
        data.elementType(), outputShape, debugContext("result"));

    setOutTensor(TiedGatherOp::outIndex(), result);
  } else {
    // Flatten the scalar indices.
    auto offsets = indices.flatten();

    // Place the gather axis at the front.
    data = data.dimShufflePartial({0}, {axis});
    // Store the shape for later.
    auto tmpShape = data.shape();
    // Flatten the other dimensions.
    data = data.flatten(1, data.rank());

    poplar::Tensor mask;
    if (op.zeroOutOfRangeIndices()) {
      std::tie(offsets, mask) =
          zeroIndiciesThatAreOutOfRange(prog, data, offsets);
    }

    offsets = offsets.reinterpret(poplar::UNSIGNED_INT);

    auto result = popops::gather(graph(),
                                 data,
                                 offsets,
                                 0,
                                 prog,
                                 popops::GatherParams(),
                                 debugContext());

    if (op.zeroOutOfRangeIndices()) {
      zeroOutputOfOutOfRangeIndices(prog, result, mask, data);
    }

    // Reshape the result to "unflatten" the other dimensions.
    tmpShape.front() = result.dim(0);
    result           = result.reshape(tmpShape);
    // Put the gather axis dimension back in the right place.
    result = result.dimShufflePartial({axis}, {0});

    // Reshape into the expected ONNX shape.
    result = result.reshape(outputShape);

    setOutTensor(TiedGatherOp::outIndex(), result);
  }
}

namespace {
OpxCreator<TiedGatherOpx>
    tiedGatherOpxCreator(Onnx::CustomOperators::TiedGather);
}

} // namespace popx
} // namespace popart
