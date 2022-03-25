// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <snap/popops/ElementWise.hpp>
#include <popops/Cast.hpp>
#include <poprand/RandomGen.hpp>

#include <popart/error.hpp>
#include <popart/ir.hpp>
#include <popart/op/dropout.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/dropoutx.hpp>
#include <popart/popx/opxmanager.hpp>

namespace pe = popops::expr;

namespace popart {
namespace popx {

namespace {

std::pair<snap::Tensor, snap::Tensor>
growDropout(snap::Graph &graph,
            const snap::Tensor &input,
            const poplar::Tensor &seed,
            const poplar::Tensor &refTensor,
            float ratio,
            const PopOpx &opx,
            snap::program::Sequence &prog) {
  double dropoutProbability = 1. - static_cast<double>(ratio);

  // When ratio is outside of (0,1), an error is thrown in op creation,
  // so we avoid div/0 errors here.
  float scale = 1.f / (1.f - ratio);

  // Calculate the dropout mask using poplibs and a tensor of ones.
  auto mask = snap::Tensor{poprand::bernoulli(graph.getPoplarGraph(),
                                              &seed,
                                              0u,
                                              refTensor,
                                              refTensor.elementType(),
                                              dropoutProbability,
                                              prog.getPoplarSequence(),
                                              opx.debugContext("mask")),
                           graph};

  // Use the mask to multiply by the input tensor and scale up.
  auto dropout =
      snap::popops::map(graph,
                        pe::Mul(pe::Mul(pe::_1, pe::_2), pe::Const(scale)),
                        {mask, input},
                        prog,
                        opx.debugContext("dropout"));

  return {dropout, mask};
}

} // namespace

DropoutOpx::DropoutOpx(Op *op, Devicex *devicex)
    : ElementWiseUnaryOpx(op, devicex) {
  verifyOp<DropoutOp>(op,
                      {Onnx::Operators::Dropout_6,
                       Onnx::Operators::Dropout_7,
                       Onnx::Operators::Dropout_10});
}

void DropoutOpx::grow(snap::program::Sequence &prog) const {
  auto &op = getOp<DropoutOp>();

  if (op_p->getIr().canTrain()) {
    const poplar::Tensor &refTensor =
        get(op.getReferenceTensorId()).getPoplarTensor();

    if (op.getOutputMask()) {
      auto dropout_mask =
          growDropout(graph(),
                      getInTensor(DropoutOp::getInIndex()),
                      getInTensor(op.getSeedInIndex()).getPoplarTensor(),
                      refTensor,
                      op.getRatio(),
                      *this,
                      prog);
      auto dropout = dropout_mask.first;
      auto mask    = dropout_mask.second;

      setOutTensor(op.getOutIndex(), dropout);
      if (op.output->hasIndex(DropoutOp::getMaskOutIndex())) {
        setOutTensor(DropoutOp::getMaskOutIndex(),
                     snap::Tensor{popops::cast(graph().getPoplarGraph(),
                                               mask.getPoplarTensor(),
                                               poplar::BOOL,
                                               prog.getPoplarSequence(),
                                               debugContext("mask")),
                                  graph()});
      }
    } else {
      double dropoutProbability = 1. - static_cast<double>(op.getRatio());
      double scale = 1. / (1. - static_cast<double>(op.getRatio()));
      auto dropout = poprand::dropout(
          graph().getPoplarGraph(),
          &getInTensor(op.getSeedInIndex()).getPoplarTensor(),
          0u,
          getInTensor(DropoutOp::getInIndex()).getPoplarTensor(),
          refTensor,
          dropoutProbability,
          scale,
          prog.getPoplarSequence(),
          debugContext("dropout"));
      setOutTensor(op.getOutIndex(), snap::Tensor{dropout, graph()});
    }
  } else {
    // In inference mode, dropout is an identity function
    auto output = cloneNcopy(prog, getInTensor(DropoutOp::getInIndex()));
    setOutTensor(DropoutOp::getOutIndex(), output);
    // In inference mask is just a tensor of true values.
    if (op.getOutputMask()) {
      auto mask = getConst(poplar::BOOL,
                           getInTensor(DropoutOp::getInIndex()).shape(),
                           true,
                           "mask");
      setOutTensor(DropoutOp::getMaskOutIndex(), mask);
    }
  }
}

InputCreatorType DropoutOpx::getInputCreatorType(InIndex inIndex) const {
  if (inIndex == DropoutOp::getInIndex()) {
    return ElementWiseUnaryOpx::getInputCreatorType(inIndex);
  }
  return PopOpx::getInputCreatorType(inIndex);
}

namespace {
OpxCreator<DropoutOpx> dropoutOpxCreator({Onnx::Operators::Dropout_6,
                                          Onnx::Operators::Dropout_7,
                                          Onnx::Operators::Dropout_10});
} // namespace

} // namespace popx
} // namespace popart
