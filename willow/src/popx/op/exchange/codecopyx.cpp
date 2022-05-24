// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <memory>
#include <popart/graphcoreoperators.hpp>
#include <popart/op/exchange/codecopy.hpp>
#include <popart/popx/op/exchange/codecopyx.hpp>
#include <popart/popx/opxmanager.hpp>

#include "popart/popx/op/exchange/exchangex.hpp"

namespace snap {
namespace program {
class Sequence;
} // namespace program
} // namespace snap

namespace popart {
class Op;

namespace popx {
class Devicex;

// ExternalCodeCopyOpx
ExternalCodeCopyOpx::ExternalCodeCopyOpx(Op *op, Devicex *devicex)
    : ExchangeBaseOpx(op, devicex) {
  verifyOp<ExternalCodeCopyOp>(op);
}

void ExternalCodeCopyOpx::grow(snap::program::Sequence &prog) const {
  auto &externalCodeCopyOp = getOp<ExternalCodeCopyOp>();
  std::shared_ptr<ExchangeDescriptorx> descriptorx =
      getExchangeDescriptorx(dv_p, externalCodeCopyOp.getExchangeDescriptor(0));

  auto debug = debugContext(logging::format(
      "ExternalCodeCopy Remote -> Device , graph: {}",
      externalCodeCopyOp.getExchangeDescriptor(0).getGraphToLoadId()));

  descriptorx->pre(graph(), prog, debug);
  descriptorx->exchange(graph(), prog, debug);
  descriptorx->post(graph(), prog, debug);
}

namespace {

OpxCreator<ExternalCodeCopyOpx>
    ExternalCodeCopyOpxCreator(Onnx::CustomOperators::ExternalCodeCopy);
} // namespace
} // namespace popx
} // namespace popart
