#include <popops/ElementWise.hpp>
#include <poprand/RandomGen.hpp>
#include <poponnx/error.hpp>
#include <poponnx/ir.hpp>
#include <poponnx/op/dropout.hpp>
#include <poponnx/popx/devicex.hpp>
#include <poponnx/popx/op/dropoutx.hpp>
#include <poponnx/popx/opxmanager.hpp>

namespace poponnx {
namespace popx {

DropoutOpx::DropoutOpx(Op *op, Devicex *devicex)
    : ElementWiseUnaryOpx(op, devicex) {
  verifyOp<DropoutOp>(op,
                      {Onnx::Operators::Dropout_6,
                       Onnx::Operators::Dropout_7,
                       Onnx::Operators::Dropout_10});

  if (dv_p->isDropoutRandomSeedRequired() == false) {
    dv_p->setDropoutRandomSeedIsRequired(true);
  }
}

void DropoutOpx::grow(poplar::program::Sequence &prog) const {
  if (op_p->getIr().canTrain()) {
    auto dropoutOp    = dynamic_cast<DropoutOp *>(op_p);
    auto seedModifier = dropoutOp->getSeedModifier();

    // Converting from poponnx standard (float) to poplar (double) for ratio
    auto ratio              = static_cast<double>(dropoutOp->getRatio());
    auto dropoutProbability = 1 - ratio;

    // If fwd dropout op, add reference tensor for layer to map.
    // If a bwd dropout op, or recomputation op, retrieve the reference
    // tensor for that layer.
    poplar::Tensor refTensor;
    if (dv_p->dropoutReferenceTensors.find(seedModifier) ==
        dv_p->dropoutReferenceTensors.end()) {
      refTensor = getInTensor(DropoutOp::getInIndex());
      dv_p->dropoutReferenceTensors.emplace(seedModifier, refTensor);
    } else {
      refTensor = dv_p->dropoutReferenceTensors.at(seedModifier);
    }

    auto seed    = getSeed(prog);
    auto dropout = poprand::dropout(graph(),
                                    &seed,
                                    seedModifier,
                                    getInTensor(DropoutOp::getInIndex()),
                                    refTensor,
                                    dropoutProbability,
                                    1 / dropoutProbability,
                                    prog,
                                    debugPrefix("dropout"));

    setOutTensor(dropoutOp->getOutIndex(), dropout);
  } else {
    // In inference/evaluation mode, dropout is an idendity function
    setOutTensor(DropoutOp::getOutIndex(),
                 getInTensor(DropoutOp::getInIndex()));
  }
}

poplar::Tensor DropoutOpx::getSeed(poplar::program::Sequence &prog) const {
  if (dv_p->getReplicationFactor() == 1) {
    return *dv_p->getDropoutRandomSeed();
  } else {
    static unsigned tileCounter = 0;
    auto tile = tileCounter % graph().getTarget().getTilesPerIPU();
    tileCounter++;

    auto indexConstant = graph().addReplicationIndexConstant();
    graph().setTileMapping(indexConstant, tile);

    auto seed = popops::map(graph(),
                            popops::expr::BinaryOpType::ADD,
                            *dv_p->getDropoutRandomSeed(),
                            indexConstant,
                            prog,
                            debugPrefix("seedAddReplicationIndex"));

    return seed;
  }
}

namespace {
OpxCreator<DropoutOpx> dropoutOpxCreator({Onnx::Operators::Dropout_6,
                                          Onnx::Operators::Dropout_7,
                                          Onnx::Operators::Dropout_10});
OpxCreator<Opx> dropoutGradOpxCreator(Onnx::GradOperators::DropoutGrad,
                                      "DropoutGradOp should be optimised out, "
                                      "\"DropoutGradOp\" pattern is required");
} // namespace

} // namespace popx
} // namespace poponnx
