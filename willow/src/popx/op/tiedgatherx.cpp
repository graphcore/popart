// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <popart/popx/op/tiedgatherx.hpp>

#include <popart/op/tiedgather.hpp>
#include <popart/opidentifier.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/irlowering.hpp>
#include <popart/popx/opxmanager.hpp>

#include <poplin/MatMul.hpp>
#include <popops/Cast.hpp>
#include <popops/DynamicSlice.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Gather.hpp>

#include <vector>

namespace popart {
namespace popx {

TiedGatherOpx::TiedGatherOpx(Op *op, Devicex *device)
    : GatherBaseOpx(op, device) {
  verifyOp<TiedGatherOp>(op, {Onnx::CustomOperators::TiedGather});

  setCommonMembersPostVerify(op);
}

InputCreatorType TiedGatherOpx::getInputCreatorType(int index0) const {
  return index0 == TiedGatherOp::dataInIndex()
             ? InputCreatorType::CanCreate
             : PopOpx::getInputCreatorType(index0);
}

snap::Tensor
TiedGatherOpx::createInputTensor(const InIndex index,
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

  return snap::Tensor{poplin::createMatMulInputRHS(graph().getPoplarGraph(),
                                                   popType(weightInfo),
                                                   lhsShape,
                                                   rhsShape,
                                                   dnai,
                                                   {},
                                                   &dv_p->matmulCache),
                      graph()};
}

// Identical to GatherOpx::grow however:
//    1) uses popops::gather instead of popops::multislice
//    2) range checks the indices and masks those out of range
void TiedGatherOpx::grow(poplar::program::Sequence &prog) const {
  const auto indicesShape = inShape(TiedGatherOp::indicesInIndex());
  const auto outputShape =
      vXtoY<int64_t, std::size_t>(outShape(TiedGatherOp::outIndex()));

  auto op       = getOp<TiedGatherOp>();
  unsigned axis = op.getAxis();
  auto indices  = getInTensor(TiedGatherOp::indicesInIndex()).getPoplarTensor();
  auto data     = getInTensor(TiedGatherOp::dataInIndex()).getPoplarTensor();

  // If there are no indices, return an empty tensor of the appropriate
  // shape
  if (indices.numElements() == 0) {
    auto result = graph().getPoplarGraph().addVariable(
        data.elementType(), outputShape, debugContext("result"));

    setOutTensor(TiedGatherOp::outIndex(), snap::Tensor{result, graph()});
  } else {
    // Flatten the scalar indices.
    auto offsets = indices.flatten();
    // reinterpret the indices as unsigned int. This assumes negative indices.
    // are impossible.
    offsets = offsets.reinterpret(poplar::UNSIGNED_INT);

    // Place the gather axis at the front.
    data = data.dimShufflePartial({0}, {axis});
    // Store the shape for later.
    auto tmp_shape = data.shape();
    // Flatten the other dimensions.
    data = data.flatten(1, data.rank());

    // Change (2)
    poplar::Tensor mask;
    if (op.checkIndices()) {
      auto gather_size  = data.shape()[0];
      mask              = popops::lt(graph().getPoplarGraph(),
                        offsets,
                        static_cast<unsigned>(gather_size),
                        prog,
                        debugContext("mask<size"));
      auto indices_mask = popops::cast(graph().getPoplarGraph(),
                                       mask,
                                       offsets.elementType(),
                                       prog,
                                       debugContext("mask_castInt"));
      offsets           = popops::mul(graph().getPoplarGraph(),
                            offsets,
                            indices_mask,
                            prog,
                            debugContext("masked_indices"));
    }

    // Change (1)
    auto result = popops::gather(graph().getPoplarGraph(),
                                 data,
                                 offsets,
                                 0,
                                 prog,
                                 popops::GatherParams(),
                                 debugContext());

    // Change (2)
    if (op.checkIndices()) {
      auto out_mask = popops::cast(graph().getPoplarGraph(),
                                   mask,
                                   data.elementType(),
                                   prog,
                                   debugContext("mask_cast"));
      popops::mulInPlace(graph().getPoplarGraph(),
                         result,
                         out_mask.expand({1}),
                         prog,
                         debugContext("masked_result"));
    }

    // Reshape the result to "unflatten" the other dimensions.
    tmp_shape.front() = result.dim(0);
    result            = result.reshape(tmp_shape);
    // Put the gather axis dimension back in the right place.
    result = result.dimShufflePartial({axis}, {0});

    // Reshape into the expected ONNX shape.
    result = result.reshape(outputShape);

    setOutTensor(TiedGatherOp::outIndex(), snap::Tensor{result, graph()});
  }
}

namespace {
OpxCreator<TiedGatherOpx>
    tiedGatherOpxCreator(Onnx::CustomOperators::TiedGather);
}

} // namespace popx
} // namespace popart
