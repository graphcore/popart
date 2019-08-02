#include <memory>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/ipucopy.hpp>
#include <popart/opmanager.hpp>
#include <popart/patterns/contiguateipucopyindices.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorindex.hpp>

namespace popart {

bool ContiguateIpuCopyIndicesPattern::matches(Op *op) const {
  auto copyOp = dynamic_cast<IpuCopyOp *>(op);
  if (copyOp) {
    if (copyOp->getSourceIpus().size() != 1) {
      return false;
    }
    auto delta = static_cast<int64_t>(copyOp->getDestIpu()) -
                 static_cast<int64_t>(copyOp->getSourceIpu());

    if (delta == +1 || delta == -1) {
      return false;
    }
    return true;
  }
  // if not a IpuCopyOp, return false
  return false;
}

std::vector<const Tensor *>
ContiguateIpuCopyIndicesPattern::touches(Op *) const {
  return {};
}

bool ContiguateIpuCopyIndicesPattern::apply(Op *op) const {

  // We assume that a call to matches has confirmed that this cast is valid
  auto originalIpuCopyOp = dynamic_cast<IpuCopyOp *>(op);

  // Creation of intermediate IpuCopyOps:
  int64_t firstIpuId = originalIpuCopyOp->getSourceIpu();
  int64_t finalIpuId = originalIpuCopyOp->getDestIpu();
  auto delta         = firstIpuId < finalIpuId ? +1 : -1;
  std::vector<std::unique_ptr<IpuCopyOp>> seq;
  for (int64_t src = firstIpuId; src != finalIpuId; src += delta) {
    auto dst = src + delta;
    seq.push_back(std::make_unique<IpuCopyOp>(
        Onnx::CustomOperators::IpuCopy, dst, op->getSettings()));
  }

  // Connect the input tensors to the firstIpuCopy of the sequence
  auto firstIpuCopy = seq.front().get();
  for (auto &tensors : op->input->tensorIdMap()) {
    firstIpuCopy->connectInTensor(
        tensors.first, tensors.second, originalIpuCopyOp->getSourceIpu());
  }

  // Connect the output tensor to the back of the sequence
  auto &finalIpuCopy = seq.back();
  if (op->output->tensorIdMap().size() != 1) {
    throw error("ILE : expected a single output Tensor for an IpuCopyOp");
  }
  for (auto &tensors : op->output->tensorIdMap()) {
    finalIpuCopy->connectOutTensor(tensors.first, tensors.second);
  }

  // Connect the sequence of IpuCopys with intermediate Tensors
  int seqIndex = 0;
  for (int64_t src = firstIpuId; src != finalIpuId - delta; src += delta) {
    auto srcOp  = seq.at(seqIndex).get();
    auto destOp = seq.at(seqIndex + 1).get();
    auto tensor = PreAliasPattern::createIntermediateTensorId(
        originalIpuCopyOp->output->id(0));
    srcOp->createAndConnectOutTensor(0, tensor);
    destOp->connectInTensor(0, tensor, src + delta);
    ++seqIndex;
  }

  // Insert the newly minted Ops into the IR
  for (auto &step : seq) {
    logging::pattern::debug(
        "Inserting IpuCopyOp {}, {}", step->str(), step->getFromToStr());
    step->setup();
    op->getGraph().moveIntoGraph(std::move(step));
  }

  // Delete the matched IpuCopyOp
  logging::pattern::info("Removing Op {}, {}",
                         op->str(),
                         dynamic_cast<IpuCopyOp *>(op)->getFromToStr());
  op->disconnectAllInputs();
  op->getGraph().eraseOp(op->id);

  logging::pattern::info("Contiguation of IpuCopyOp is complete");
  return true;
}

namespace {
static PatternCreator<ContiguateIpuCopyIndicesPattern>
    contiguateIpuCopyIndicesPattern(
        PreAliasPatternType::CONTIGUATEIPUCOPYINDICES,
        "ContiguateIpuCopyIndicesPattern");
}

} // namespace popart
