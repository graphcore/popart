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

// Get the reference Tensor used for poplibs call for mask generation.
// Note that the reference Tensor cannot just be the Tensor to be masked, as the
// mask generated depends on the layout of the input and not just the random
// seed. It is required that forwards, recompute and backwards masks are all the
// same, so the same reference tensor must be used in for these cases
poplar::Tensor getReferenceTensor(const Opx &opx) {

  const auto &dbo   = opx.getOp<DropoutOp>();
  auto seedModifier = dbo.getSeedModifier();

  poplar::Tensor refTensor;

  if (opx.getDropoutReferenceTensors().find(seedModifier) ==
      opx.getDropoutReferenceTensors().end()) {
    refTensor = opx.getInTensor(0);
    opx.getDropoutReferenceTensors().emplace(seedModifier, refTensor);
  } else {
    refTensor = opx.getDropoutReferenceTensors().at(seedModifier);
  }
  return refTensor;
}

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
                                 opx.debugPrefix("mask"));

  // Use the mask to multiply by the input tensor and scale up.
  auto dropout = popops::map(graph,
                             pe::Mul(pe::Mul(pe::_1, pe::_2), pe::Const(scale)),
                             {mask, input},
                             prog,
                             opx.debugPrefix("dropout"));

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
    poplar::Tensor refTensor = getReferenceTensor(*this);

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
                graph(), mask, poplar::BOOL, prog, debugPrefix("mask")));
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
                                      debugPrefix("dropout"));
      setOutTensor(op.getOutIndex(), dropout);
    }
  } else {
    // In inference/evaluation mode, dropout is an identity function
    setOutTensor(DropoutOp::getOutIndex(),
                 getInTensor(DropoutOp::getInIndex()));
    // In inference mask is just a tensor of true values.
    if (op.getOutputMask()) {
      auto mask = getConst(poplar::BOOL,
                           getInTensor(DropoutOp::getInIndex()).shape(),
                           true,
                           debugPrefix("mask"));
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
