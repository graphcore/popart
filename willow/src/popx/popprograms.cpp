// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <cctype>
#include <fstream>
#include <random>
#include <set>

#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/popprograms.hpp>

namespace popart {
namespace popx {

PopPrograms::PopPrograms(Devicex *dv_p_) : dv_p(dv_p_) {}

const poplar::program::Sequence &
PopPrograms::streamWeightsFromHostFragment() const {
  return seqs[static_cast<int>(ProgramFragmentIndex::StreamWeightsFromHost)];
}
poplar::program::Sequence &PopPrograms::streamWeightsFromHostFragment() {
  return seqs[static_cast<int>(ProgramFragmentIndex::StreamWeightsFromHost)];
}

const poplar::program::Sequence &
PopPrograms::streamOptimizerFromHostFragment() const {
  return seqs[static_cast<int>(ProgramFragmentIndex::StreamOptimizerFromHost)];
}
poplar::program::Sequence &PopPrograms::streamOptimizerFromHostFragment() {
  return seqs[static_cast<int>(ProgramFragmentIndex::StreamOptimizerFromHost)];
}

const poplar::program::Sequence &
PopPrograms::setRandomSeedFromHostFragment() const {
  return seqs[static_cast<int>(ProgramFragmentIndex::SetRandomSeedFromHost)];
}
poplar::program::Sequence &PopPrograms::setRandomSeedFromHostFragment() {
  return seqs[static_cast<int>(ProgramFragmentIndex::SetRandomSeedFromHost)];
}

const poplar::program::Sequence &
PopPrograms::cycleCountTensorToHostFragment() const {
  return seqs[static_cast<int>(ProgramFragmentIndex::CycleCountTensortoHost)];
}
poplar::program::Sequence &PopPrograms::cycleCountTensorToHostFragment() {
  return seqs[static_cast<int>(ProgramFragmentIndex::CycleCountTensortoHost)];
}

const poplar::program::Sequence &PopPrograms::initFragment() const {
  return seqs[static_cast<int>(ProgramFragmentIndex::Init)];
}

poplar::program::Sequence &PopPrograms::initFragment() {
  return seqs[static_cast<int>(ProgramFragmentIndex::Init)];
}

const poplar::program::Sequence &PopPrograms::preForwardFragment() const {
  return seqs[static_cast<int>(ProgramFragmentIndex::PreForward)];
}

poplar::program::Sequence &PopPrograms::preForwardFragment() {
  return seqs[static_cast<int>(ProgramFragmentIndex::PreForward)];
}

const poplar::program::Sequence &PopPrograms::forwardFragment() const {
  return seqs[static_cast<int>(ProgramFragmentIndex::Forward)];
}

poplar::program::Sequence &PopPrograms::forwardFragment() {
  return seqs[static_cast<int>(ProgramFragmentIndex::Forward)];
}

const poplar::program::Sequence &PopPrograms::backwardFragment() const {
  return seqs[static_cast<int>(ProgramFragmentIndex::Backward)];
}

poplar::program::Sequence &PopPrograms::backwardFragment() {
  return seqs[static_cast<int>(ProgramFragmentIndex::Backward)];
}

const poplar::program::Sequence &PopPrograms::toHostFinalCopyFragment() const {
  return seqs[static_cast<int>(ProgramFragmentIndex::ToHostFinalCopy)];
}

poplar::program::Sequence &PopPrograms::toHostFinalCopyFragment() {
  return seqs[static_cast<int>(ProgramFragmentIndex::ToHostFinalCopy)];
}

const poplar::program::Sequence &PopPrograms::accumulateOuterFragment() const {
  return seqs[static_cast<int>(ProgramFragmentIndex::VarUpdateFromAccumulator)];
}

poplar::program::Sequence &PopPrograms::accumulateOuterFragment() {
  return seqs[static_cast<int>(ProgramFragmentIndex::VarUpdateFromAccumulator)];
}

const poplar::program::Sequence &PopPrograms::weightsToHostFragment() const {
  return seqs[static_cast<int>(ProgramFragmentIndex::WeightstoHost)];
}

poplar::program::Sequence &PopPrograms::weightsToHostFragment() {
  return seqs[static_cast<int>(ProgramFragmentIndex::WeightstoHost)];
}

poplar::program::Sequence PopPrograms::weightsFromHost() const {
  poplar::program::Sequence prog;
  prog.add(streamWeightsFromHostFragment());
  return prog;
}

poplar::program::Sequence PopPrograms::optimizerFromHost() const {
  poplar::program::Sequence prog;
  prog.add(streamOptimizerFromHostFragment());
  return prog;
}

poplar::program::Sequence PopPrograms::setRandomSeedFromHost() const {
  poplar::program::Sequence prog;
  prog.add(setRandomSeedFromHostFragment());
  return prog;
}

poplar::program::Sequence PopPrograms::cycleCountTensorToHost() const {
  poplar::program::Sequence prog;
  prog.add(cycleCountTensorToHostFragment());
  return prog;
}

void PopPrograms::addPipelineCycle(
    PipelineCycle pCycle,
    poplar::program::Sequence &sq,
    std::ostringstream &ss,
    std::map<PipelineStage, poplar::Function> &fwdFunctions) const {
  // Inside the each phase, conditionally do:
  //
  // 1. The pre-forward fragment
  // 2. Host->Device copies for each IPU
  // 3. Forward fragments for each IPU
  // 7. Device->Host copies for each IPU
  // 8. Inter-IPU copies

  PipelineInfo pInfo = dv_p->pipelineInfo();

  // 1.
  sq.add(preForwardFragment());

  // 2.
  if (pipelineSeqs.find(PipelineFragmentId::ToDeviceStream) !=
      pipelineSeqs.end()) {
    for (auto &stage_seq :
         pipelineSeqs.at(PipelineFragmentId::ToDeviceStream)) {
      if (pInfo.doStage(pCycle, stage_seq.first)) {
        ss << "\n  ps" << stage_seq.first << " : ToDeviceStream";
        sq.add(stage_seq.second);
      }
    }
  } else {
    if (dv_p->ir().useSyntheticData() == false) {
      throw error(
          "There are no ToDeviceStream pipeline program fragments. Check that "
          "the stream copies have been added to the correct fragment.");
    }
  }

  auto tryAddRestoreFragmentForStage = [&](PipelineStage stage) {
    auto foundFragment = pipelineSeqs.find(PipelineFragmentId::Restore);
    if (foundFragment != pipelineSeqs.end()) {
      auto &stages_seqs = foundFragment->second;
      auto foundStage   = stages_seqs.find(stage);
      if (foundStage != stages_seqs.end()) {
        auto &sequence = foundStage->second;
        ss << "\n  ps" << stage << " : Restore";
        sq.add(sequence);
      }
    }
  };

  // 3.
  for (auto &stage_seq : fwdFunctions) {
    auto stage = stage_seq.first;
    if (pInfo.doStage(pCycle, stage)) {
      tryAddRestoreFragmentForStage(stage);
      ss << "\n  ps" << stage << " : Forward";
      sq.add(poplar::program::Call(stage_seq.second));
    }
  }

  // 7.
  if (pipelineSeqs.find(PipelineFragmentId::ToHostStream) !=
      pipelineSeqs.end()) {
    for (auto &stage_seq : pipelineSeqs.at(PipelineFragmentId::ToHostStream)) {
      if (pInfo.doStage(pCycle, stage_seq.first)) {
        ss << "\n  ps" << stage_seq.first << " : ToHostStream";
        sq.add(stage_seq.second);
      }
    }
  }

  // Insert the IPU-copies.
  // Note: Always do all the copies. This is ensure that ALL copies are
  // outlined across pipelineCycles AND merged across pipelineStages.
  ss << logging::format("\n  IpuCopies");
  sq.add(pipelineIpuCopySeq);
}

poplar::program::Sequence
PopPrograms::getMainProgramFromPipelineFragments() const {

  // What's happening here? Consider the model:
  //
  // A  A' IPU0
  // |  |
  // B  B' IPU1
  // |  |
  // C--C' IPU2     (where X' is the grad op of X)
  //
  // The schedule on each IPU looks as follows:
  // 1. F<i> - execute fwd ops on batch i on the IPU, and stash activations
  // 2. B<i> - restore activations and run bwd ops on batch i on the IPU
  // 3. C<i>F - copy activations for batch i to the next IPU
  // 4. C<i>B - copy grads for batch i to the previous IPU
  //
  // A pipeline with the minimum number of steps for 3 IPUs looks as follows:
  //
  // Training mode:
  //
  // clang-format off
  //
  //       <- fwd fill -> <-------- bwd fill --------> <--- main ---> <------ fwd flush ------> < bwd flush ->
  // IPU0: F0.C0F, F1.C1F, F2.   C2F   , F3.C3F       , F4.B0.C0F    ,    B1        ,    B2    , B3    , B4
  // IPU1:         F0.C0F, F1.   C1F   , F2.B0.C2F.C0B, F3.B1.C3F.C1B, F4.B2.C4F.C2B,    B3.C3B, B4.C4B,
  // IPU2:                 F0.B0.   C0B, F1.B1.    C1B, F2.B2.    C2B, F3.B3.   .C3B, F4.B4.C4B,
  //
  // clang-format on
  //
  // Inference mode:
  //
  //       <- fwd fill -> < main> <fwd flush>
  // IPU0: F0.C0F, F1.C1F, F2.C2F,
  // IPU1:         F0.C0F, F1.C1F, F2.C2F,
  // IPU2:                 F0.   , F1.   , F2
  //
  // The contents of the program inside the repeat loop should resemble
  // a vertical slice of the the above diagrams (i.e. one pipeline cycle).
  // To achieve the 'holes' in the fwd/bwd fill and fwd/bwd flush cycles,
  // we run the fwd and bwd programs inside a conditional 'If' program.
  //
  // Note that the IPU copies are always run regardless of the conditionals.
  // Host<->Device streams are run as normal with a the fwd/bwd program
  // fragments.

  // Which parts of the Ir graph are run in each of the pipeline
  // fragments? Print this info here:
  std::ostringstream ss;
  ss << "\n";
  for (auto &fragid_ipudescs : pipelineDescs) {
    PipelineFragmentId fragId = fragid_ipudescs.first;
    std::string fragStr       = getStrFromPipelineFragmentId(fragId);
    ss << "\n" + fragStr + ":";
    for (auto stage_desc : pipelineDescs.at(fragId)) {
      auto vgStr = std::to_string(stage_desc.first);
      auto desc  = stage_desc.second;
      ss << "\n  ps" + vgStr + ":" + desc;
    }
  }

  ss << logging::format("\nIpuCopies: {}", pipelineIpuCopyDesc);

  ss << "\n\n";

  PipelineInfo pInfo = dv_p->pipelineInfo();

  std::map<PipelineStage, poplar::Function> fwdFunctions;

  for (auto &stage_seq : pipelineSeqs.at(PipelineFragmentId::Forward)) {
    fwdFunctions.insert(
        {stage_seq.first, dv_p->graph().addFunction(stage_seq.second)});
  }

  poplar::program::Sequence fill;
  for (PipelineCycle pCycle = pInfo.fillPhase.start;
       pCycle <= pInfo.fillPhase.end;
       pCycle++) {
    ss << "\nPipeline Cycle " + std::to_string(pCycle) + ":";
    addPipelineCycle(pCycle, fill, ss, fwdFunctions);
  }

  // All pipeline cycles in the main phase are identical. So we create the
  // program for a single cycle and repeat for mainCycles
  poplar::program::Sequence main;
  int64_t mainCycles = pInfo.mainPhase.end - pInfo.mainPhase.start + 1;
  ss << "\nPipeline Cycle 'Main', " + std::to_string(mainCycles) + " cycles";
  addPipelineCycle(pInfo.mainPhase.start, main, ss, fwdFunctions);

  poplar::program::Sequence flush;
  for (PipelineCycle pCycle = pInfo.flushPhase.start;
       pCycle <= pInfo.flushPhase.end;
       pCycle++) {
    ss << "\nPipeline Cycle " + std::to_string(pCycle) + ":";
    addPipelineCycle(pCycle, flush, ss, fwdFunctions);
  }

  logging::devicex::debug("Pipelining program construction summary:");
  logging::devicex::debug(ss.str());

  poplar::program::Sequence inner;

  inner.add(initFragment());
  inner.add(fill);
  // This is the inner main cycles loop, if doing pipelining withour gradient
  // accumulation, this the batches per step loop, as batch size = micro_batch
  // size
  inner.add(poplar::program::Repeat(static_cast<uint32_t>(mainCycles), main));
  inner.add(flush);
  poplar::program::Sequence outer;

  if (!dv_p->getOuterLoopFragEmpty()) {

    inner.add(accumulateOuterFragment());
    // If doing gradient accumulation, the inner loop is over mini batches,
    // and this outer loop loops over multiple batches per step.
    auto bps = dv_p->ir().getDataFlow().batchesPerStep();
    outer    = poplar::program::Repeat(bps, inner);
  } else {
    // No gradient accumulation, so just add one iteration of the inner program.
    outer.add(inner);
  }

  outer.add(toHostFinalCopyFragment());

  return outer;
}

poplar::program::Sequence PopPrograms::program() const {
  auto instrumentations =
      dv_p->ir().getSessionOptions().hardwareInstrumentations;

  poplar::program::Sequence outer;
  if (dv_p->ir().getSessionOptions().enablePipelining) {
    outer.add(getMainProgramFromPipelineFragments());
  } else {
    poplar::program::Sequence prog;
    prog.add(preForwardFragment());
    prog.add(forwardFragment());
    prog.add(backwardFragment());

    outer.add(initFragment());

    // auto accumulationFactor = static_cast<int>(
    auto accumulationFactor = dv_p->getAccumulationFactor();
    if (!dv_p->getOuterLoopFragEmpty()) {
      logging::devicex::trace(
          "Adding gradient accumulation repeat loop with {} loops",
          accumulationFactor);
      prog = poplar::program::Repeat(accumulationFactor, prog);
      prog.add(accumulateOuterFragment());
    }

    if (dv_p->ir().getSessionOptions().instrumentWithHardwareCycleCounter &&
        instrumentations.find(Instrumentation::Inner) !=
            instrumentations.end()) {
      // Instrument first tile of every IPU for inner program
      for (size_t i = 0; i < dv_p->getDeviceInfo()->getNumIpus(); ++i) {
        std::stringstream ss;
        // String to identify instrumentation
        ss << "inner_ipu_" << i;
        dv_p->instrumentWithHardwareCycleCounter(
            prog, i * dv_p->getDeviceInfo()->getTilesPerIpu(), ss.str());
      }
    }

    // BatchesPerStep loop
    outer.add(poplar::program::Repeat(dv_p->ir().getDataFlow().batchesPerStep(),
                                      prog));
    outer.add(toHostFinalCopyFragment());
  }

  if (dv_p->ir().getSessionOptions().instrumentWithHardwareCycleCounter &&
      instrumentations.find(Instrumentation::Outer) != instrumentations.end()) {
    dv_p->instrumentWithHardwareCycleCounter(outer);
  }

  return outer;
}

poplar::program::Sequence PopPrograms::weightsToHost() const {
  return weightsToHostFragment();
}

const std::vector<poplar::program::Program> PopPrograms::progs() const {
  std::vector<poplar::program::Program> ps(ProgramIndex::N);

  ps[ProgramIndex::WeightsFromHost]        = weightsFromHost();
  ps[ProgramIndex::OptimizerFromHost]      = optimizerFromHost();
  ps[ProgramIndex::SetRandomSeedFromHost]  = setRandomSeedFromHost();
  ps[ProgramIndex::Program]                = program();
  ps[ProgramIndex::WeightstoHost]          = weightsToHost();
  ps[ProgramIndex::CycleCountTensortoHost] = cycleCountTensorToHost();

  return ps;
}

poplar::program::Sequence &
PopPrograms::programFragment(PopPrograms::ProgramFragmentIndex index) {
  return seqs[static_cast<int>(index)];
}

poplar::program::Sequence &PopPrograms::scopeFragment(const Graph &graph) {
  if (graph.id.str().empty()) {
    throw error("There is no scope fragment for the main scope");
  } else {
    return scopeSeqs.at(graph.id.str());
  }
}

bool PopPrograms::containsFragment(const Graph &graph) const {
  if (graph.id.str().empty()) {
    return true;
  } else {
    return scopeSeqs.find(graph.id.str()) != scopeSeqs.end();
  }
}

void PopPrograms::createFragment(const Graph &graph) {
  scopeSeqs.insert({graph.id.str(), {}});
}

poplar::Function &PopPrograms::getFragmentFunction(const Graph &called_graph,
                                                   poplar::Graph &popgraph) {
  if (funcs.find(called_graph.id.str()) == funcs.end()) {
    auto &fragment = scopeFragment(called_graph);
    logging::devicex::trace("[getFragmentFunction] Creating function "
                            "for graph {}",
                            called_graph.id.str());
    funcs.insert({called_graph.id.str(), popgraph.addFunction(fragment)});
  }
  return funcs.at(called_graph.id.str());
}

bool PopPrograms::hasBeenRecomputed(OpId id, ExecutionPhase phase) const {
  auto itHas = (beenRecomputed.find({id, phase}) != beenRecomputed.end());
  return itHas;
}

void PopPrograms::recordRecomputed(OpId id, ExecutionPhase phase) {
  beenRecomputed.insert({id, phase});
}

poplar::program::Sequence &PopPrograms::recomputeFragment(OpId id) {
  auto found = recomputeSeqs.find(id);
  if (found != recomputeSeqs.end()) {
    return found->second;
  }
  recomputeSeqs.insert({id, poplar::program::Sequence{}});
  return recomputeSeqs[id];
}

poplar::program::Sequence &
PopPrograms::forwardOrBackwardFragment(ScheduledPreLoss preLoss) {
  switch (preLoss) {
  case ScheduledPreLoss::Yes: {
    return forwardFragment();
  }
  case ScheduledPreLoss::No: {
    return backwardFragment();
  }
  case ScheduledPreLoss::Undefined: {
    throw error("There is no fragment for Undefined SchedulePreLoss");
  }
  default:
    throw error("Unknown SchedulePreLoss fragment");
  }
}

poplar::program::Sequence &
PopPrograms::pipelineFragment(PipelineStage pipelineStage,
                              PipelineFragmentId frag,
                              const std::string &desc) {
  auto foundFrag = pipelineSeqs.find(frag);
  if (foundFrag != pipelineSeqs.end()) {
    auto foundPipelineStage = pipelineSeqs.at(frag).find(pipelineStage);
    if (foundPipelineStage != pipelineSeqs.at(frag).end()) {
      pipelineDescs.at(frag).at(pipelineStage).append("\n    " + desc);
      return pipelineSeqs.at(frag).at(pipelineStage);
    } else {
      pipelineDescs.at(frag).insert({pipelineStage, "\n    " + desc});
      pipelineSeqs.at(frag).insert(
          {pipelineStage, poplar::program::Sequence{}});
      return pipelineSeqs.at(frag).at(pipelineStage);
    }
  } else {
    pipelineDescs.insert({frag, {{pipelineStage, "\n    " + desc}}});
    pipelineSeqs.insert({frag, {{pipelineStage, poplar::program::Sequence{}}}});
    return pipelineSeqs.at(frag).at(pipelineStage);
  }
}

poplar::program::Sequence &
PopPrograms::pipelineRestoreFragment(PipelineStage pipelineStage,
                                     const std::string &desc) {
  return pipelineFragment(pipelineStage, PipelineFragmentId::Restore, desc);
}

poplar::program::Sequence &
PopPrograms::pipelineForwardFragment(PipelineStage pipelineStage,
                                     const std::string &desc) {
  return pipelineFragment(pipelineStage, PipelineFragmentId::Forward, desc);
}

poplar::program::Sequence &
PopPrograms::pipelineToDeviceStreamFragment(PipelineStage pipelineStage,
                                            const std::string &desc) {
  return pipelineFragment(
      pipelineStage, PipelineFragmentId::ToDeviceStream, desc);
}

poplar::program::Sequence &
PopPrograms::pipelineToHostStreamFragment(PipelineStage pipelineStage,
                                          const std::string &desc) {
  return pipelineFragment(
      pipelineStage, PipelineFragmentId::ToHostStream, desc);
}

poplar::program::Sequence &
PopPrograms::pipelineIpuCopyFragment(const std::string &desc) {
  pipelineIpuCopyDesc.append("\n    " + desc);
  return pipelineIpuCopySeq;
}

std::string
PopPrograms::getStrFromPipelineFragmentId(PipelineFragmentId fragId) const {
  switch (fragId) {
  case PipelineFragmentId::ToDeviceStream: {
    return "ToDeviceStream";
  }
  case PipelineFragmentId::Restore: {
    return "Restore";
  }
  case PipelineFragmentId::Forward: {
    return "Forward";
  }
  case PipelineFragmentId::ToHostStream: {
    return "ToHostStream";
  }
  default:
    throw error("Cannot return string for PipelineFragmentId '{}'",
                static_cast<int>(fragId));
  }
}

} // namespace popx
} // namespace popart
