#include <memory>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/ipucopy.hpp>
#include <popart/opmanager.hpp>
#include <popart/patterns/contiguateipucopyindices.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorindex.hpp>
#include <popart/topocons.hpp>

namespace popart {

bool ContiguateIpuCopyIndicesPattern::matches(Op *op) const {
  auto copyOp = dynamic_cast<IpuCopyOp *>(op);
  if (copyOp) {
    // copies of optimizer tensors run outside the main program fragment
    if (copyOp->copiesOptimizerTensors()) {
      return false;
    }

    if (copyOp->getSourceTensors().size() != 1) {
      return false;
    }

    auto out0       = copyOp->outTensor(0);
    auto firstStage = op->getPipelineStage();
    auto lastStage  = *out0->consumers.findLowestPipelineStage();
    auto delta      = lastStage - firstStage;

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

namespace {

void setPipelineStagesForSequence(std::vector<std::unique_ptr<IpuCopyOp>> &seq,
                                  PipelineStage originalOpsPipelineStage) {
  // Pipeline stages should increment/decrement over the sequence
  for (int i = 0; i < seq.size(); i++) {
    seq.at(i)->setPipelineStage(originalOpsPipelineStage + i);
  }
}

std::map<PipelineStage, VGraphId>
getPipelineStageToVGraphMap(const Graph &graph) {
  std::map<PipelineStage, VGraphId> result;
  for (auto &id_op : graph.getOps()) {
    auto op = id_op.second.get();
    if (op->hasPipelineStage() && op->hasVirtualGraphId() &&
        !op->isIpuCopyOp()) {
      logging::pattern::trace("[getPipelineStageToVGraphMap]:"
                              "Op {}, pipeline stage: {}, VGID: {}",
                              op->debugName(),
                              op->getPipelineStage(),
                              op->getVirtualGraphId());
      result[op->getPipelineStage()] = op->getVirtualGraphId();
    }
  }
  return result;
}

} // namespace

bool ContiguateIpuCopyIndicesPattern::apply(Op *op) const {

  // We assume that a call to matches has confirmed that this cast is valid
  auto originalIpuCopyOp = dynamic_cast<IpuCopyOp *>(op);

  // Creation of intermediate IpuCopyOps:
  auto out0       = originalIpuCopyOp->outTensor(0);
  auto firstStage = op->getPipelineStage();
  auto lastStage  = *out0->consumers.findLowestPipelineStage();

  auto pipelineStageToVGraph = getPipelineStageToVGraphMap(op->getGraph());

  if (lastStage <= firstStage) {
    throw internal_error(
        "Copy op {}, goes from stage {} to stage {}. Copies must always "
        "be to a later pipeline stage",
        originalIpuCopyOp->debugName(),
        firstStage,
        lastStage);
  }

  std::vector<std::unique_ptr<IpuCopyOp>> seq;
  std::vector<Op *> newIpuCopyOps;
  for (PipelineStage srcPipeline = firstStage; srcPipeline != lastStage;
       srcPipeline++) {
    auto dstPipeline = srcPipeline + 1;
    auto dst         = pipelineStageToVGraph.at(dstPipeline);
    seq.push_back(std::make_unique<IpuCopyOp>(
        Onnx::CustomOperators::IpuCopy, dst, op->getSettings()));
    newIpuCopyOps.push_back(seq.back().get());
  }

  // transfer all topological constraints to the new IpuCopy Ops
  Graph &graph = op->getGraph();
  graph.topoCons->transferToMultiple(op, newIpuCopyOps);

  setPipelineStagesForSequence(seq, originalIpuCopyOp->getPipelineStage());

  // Connect the input tensors to the firstIpuCopy of the sequence
  auto firstIpuCopy = seq.front().get();
  for (auto &tensors : op->input->tensorIdMap()) {
    firstIpuCopy->connectInTensor(
        tensors.first, tensors.second, originalIpuCopyOp->getSourceIpu());
  }

  // Connect the sequence of IpuCopys with intermediate Tensors
  for (int seqIndex = 0; seqIndex < seq.size() - 1; seqIndex++) {
    auto srcOp  = seq.at(seqIndex).get();
    auto destOp = seq.at(seqIndex + 1).get();
    for (int i = 0; i < originalIpuCopyOp->output->n(); i++) {
      auto tensor =
          createIntermediateTensorId(originalIpuCopyOp->output->id(i));
      srcOp->createAndConnectOutTensor(i, tensor);
      destOp->connectInTensor(i, tensor, srcOp->getDestIpu());
    }
  }

  // Connect the output tensor to the back of the sequence. Note: this
  // can have more than one ouput (all with the same destIPUs) if this
  // IpuCopy has been merged in the MergeCopies transform
  auto outputs = op->output->tensorIdMap();
  op->disconnectAllOutputs();
  auto &finalIpuCopy = seq.back();
  for (auto &tensors : outputs) {
    finalIpuCopy->connectOutTensor(tensors.first, tensors.second);
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
// Not registering this pattern with PatternCreator, as it is run as part of the
// Pipeline transform.
static AddPatternName<ContiguateIpuCopyIndicesPattern>
    registerName("ContiguateIpuCopyIndicesPattern");
} // namespace

} // namespace popart
