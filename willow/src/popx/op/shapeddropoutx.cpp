// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <poprand/RandomGen.hpp>

#include <popart/error.hpp>
#include <popart/ir.hpp>
#include <popart/op/shapeddropout.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/shapeddropoutx.hpp>
#include <popart/popx/opxmanager.hpp>

namespace popart {
namespace popx {

ShapedDropoutOpx::ShapedDropoutOpx(Op *op, Devicex *devicex)
    : Opx(op, devicex) {
  verifyOp<ShapedDropoutOp>(op, {Onnx::CustomOperators::ShapedDropout_1});
}

void ShapedDropoutOpx::grow(poplar::program::Sequence &prog) const {
  if (!op_p->getIr().canTrain()) {
    // In inference mode, shaped dropout is an identity function
    auto output = cloneNcopy(prog, getInTensor(ShapedDropoutOp::getInIndex()));
    setOutTensor(ShapedDropoutOp::getOutIndex(), output);
    return;
  }

  auto &op                 = getOp<ShapedDropoutOp>();
  auto seedModifier        = op.getSeedModifier();
  poplar::Tensor refTensor = getReferenceTensor();
  double keepProbability   = 1. - static_cast<double>(op.getRatio());
  double scale             = 1. / keepProbability;
  auto shapedDropout =
      poprand::shapedDropout(graph(),
                             &getInTensor(op.getSeedInIndex()),
                             seedModifier,
                             getInTensor(ShapedDropoutOp::getInIndex()),
                             refTensor,
                             keepProbability,
                             scale,
                             prog,
                             debugPrefix("shapedDropout"));

  setOutTensor(op.getOutIndex(), shapedDropout);
}

// Get the reference Tensor used for poplibs call for mask generation.
// Note that poprand uses a combination of tile-id, thread-id, seed and seed
// modifier to determine the PRNG stream. A linear mapping is always used for
// the reference tensor to support repeatable shaped dropout masks.
poplar::Tensor ShapedDropoutOpx::getReferenceTensor() const {
  const auto &dbo   = getOp<ShapedDropoutOp>();
  auto poplarType   = popType(inInfo(dbo.getInIndex()));
  auto dropoutShape = vXtoY<int64_t, std::size_t>(dbo.getShape());

  return graph().addVariable(poplarType,
                             dropoutShape,
                             poplar::VariableMappingMethod::LINEAR,
                             debugPrefix("dropoutShape"));
}

namespace {
OpxCreator<ShapedDropoutOpx>
    shapedDropoutOpxCreator({Onnx::CustomOperators::ShapedDropout_1});
} // namespace

} // namespace popx
} // namespace popart
