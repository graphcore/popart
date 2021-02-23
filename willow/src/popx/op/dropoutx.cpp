// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <popops/Cast.hpp>
#include <popops/ElementWise.hpp>
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

std::pair<poplar::Tensor, poplar::Tensor>
growDropout(poplar::Graph &graph,
            const poplar::Tensor &input,
            const poplar::Tensor &seed,
            const poplar::Tensor &refTensor,
            float ratio,
            uint32_t seedModifier,
            const Opx &opx,
            poplar::program::Sequence &prog) {
  double dropoutProbability = 1. - static_cast<double>(ratio);

  // When ratio is outside of (0,1), an error is thrown in op creation,
  // so we avoid div/0 errors here.
  float scale = 1.f / (1.f - ratio);

  // Calculate the dropout mask using poplibs and a tensor of ones.
  auto mask = poprand::bernoulli(graph,
                                 &seed,
                                 seedModifier,
                                 refTensor,
                                 refTensor.elementType(),
                                 dropoutProbability,
                                 prog,
                                 opx.debugContext("mask"));

  // Use the mask to multiply by the input tensor and scale up.
  auto dropout = popops::map(graph,
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

void DropoutOpx::grow(poplar::program::Sequence &prog) const {
  auto &op          = getOp<DropoutOp>();
  auto seedModifier = op.getSeedModifier();

  if (op_p->getIr().canTrain()) {
    const poplar::Tensor &refTensor = get(op.getReferenceTensorId());

    if (op.getOutputMask()) {
      auto dropout_mask = growDropout(graph(),
                                      getInTensor(DropoutOp::getInIndex()),
                                      getInTensor(op.getSeedInIndex()),
                                      refTensor,
                                      op.getRatio(),
                                      seedModifier,
                                      *this,
                                      prog);
      auto dropout      = dropout_mask.first;
      auto mask         = dropout_mask.second;

      setOutTensor(op.getOutIndex(), dropout);
      if (op.output->hasIndex(DropoutOp::getMaskOutIndex())) {
        setOutTensor(
            DropoutOp::getMaskOutIndex(),
            popops::cast(
                graph(), mask, poplar::BOOL, prog, debugContext("mask")));
      }
    } else {
      double dropoutProbability = 1. - static_cast<double>(op.getRatio());
      double scale = 1. / (1. - static_cast<double>(op.getRatio()));
      auto dropout = poprand::dropout(graph(),
                                      &getInTensor(op.getSeedInIndex()),
                                      seedModifier,
                                      getInTensor(DropoutOp::getInIndex()),
                                      refTensor,
                                      dropoutProbability,
                                      scale,
                                      prog,
                                      debugContext("dropout"));
      setOutTensor(op.getOutIndex(), dropout);
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

namespace {
OpxCreator<DropoutOpx> dropoutOpxCreator({Onnx::Operators::Dropout_6,
                                          Onnx::Operators::Dropout_7,
                                          Onnx::Operators::Dropout_10});
} // namespace

} // namespace popx
} // namespace popart
