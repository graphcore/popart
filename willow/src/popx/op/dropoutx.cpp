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
  auto &op = getOp<DropoutOp>();

  if (op_p->getIr().canTrain()) {
    auto seedModifier = op.getSeedModifier();

    // If fwd dropout op, add reference tensor for layer to map.
    // If a recomputation op, retrieve the reference
    // tensor for that layer.
    // Ref tensor can not be an op input because the tensor layout changes the
    // result of poprand::bernoulli, and if the op is pipelined, the stash and
    // restore ops may change the tensor layout of the ops input.
    poplar::Tensor refTensor;
    if (dv_p->dropoutReferenceTensors.find(seedModifier) ==
        dv_p->dropoutReferenceTensors.end()) {
      refTensor = getInTensor(DropoutOp::getInIndex());
      dv_p->dropoutReferenceTensors.emplace(seedModifier, refTensor);
    } else {
      refTensor = dv_p->dropoutReferenceTensors.at(seedModifier);
    }

    auto seed = cloneNcopy(prog, dv_p->getRandomSeedTensor());

    auto dropout_mask = growDropout(graph(),
                                    getInTensor(DropoutOp::getInIndex()),
                                    seed,
                                    refTensor,
                                    op.getRatio(),
                                    seedModifier,
                                    *this,
                                    prog);
    auto dropout      = dropout_mask.first;
    auto mask         = dropout_mask.second;

    setOutTensor(op.getOutIndex(), dropout);
    if (op.returnMask()) {
      setOutTensor(
          DropoutOp::getMaskOutIndex(),
          popops::cast(graph(), mask, poplar::BOOL, prog, debugPrefix("mask")));
    }
    setOutTensor(op.getSeedOutIndex(), seed);
  } else {
    // In inference/evaluation mode, dropout is an identity function
    setOutTensor(DropoutOp::getOutIndex(),
                 getInTensor(DropoutOp::getInIndex()));
    // In inference mask is just a tensor of true values.
    if (op.returnMask()) {
      auto mask =
          graph().addConstant(poplar::BOOL,
                              getInTensor(DropoutOp::getInIndex()).shape(),
                              true,
                              debugPrefix("mask"));
      setOutTensor(DropoutOp::getMaskOutIndex(), mask);
    }
  }
}

DropoutGradOpx::DropoutGradOpx(Op *op, Devicex *devicex)
    : ElementWiseUnaryOpx(op, devicex) {
  verifyOp<DropoutOp>(op, {Onnx::GradOperators::DropoutGrad});
}

void DropoutGradOpx::grow(poplar::program::Sequence &prog) const {
  auto &op          = getOp<DropoutGradOp>();
  auto seedModifier = op.getSeedModifier();

  // Fwd dropout op should have added a reference tensor.
  // See comment in forward op for why this can not be an op input.
  poplar::Tensor refTensor = dv_p->dropoutReferenceTensors.at(seedModifier);

  auto dropout_mask = growDropout(graph(),
                                  getInTensor(DropoutGradOp::getGradInIndex()),
                                  getInTensor(op.getSeedInIndex()),
                                  refTensor,
                                  op.getRatio(),
                                  seedModifier,
                                  *this,
                                  prog);
  auto dropout      = dropout_mask.first;

  setOutTensor(op.getOutIndex(), dropout);
}

namespace {
OpxCreator<DropoutOpx> dropoutOpxCreator({Onnx::Operators::Dropout_6,
                                          Onnx::Operators::Dropout_7,
                                          Onnx::Operators::Dropout_10});
OpxCreator<DropoutGradOpx>
    dropoutGradOpxCreator({Onnx::GradOperators::DropoutGrad});
} // namespace

} // namespace popx
} // namespace popart
