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

poplar::program::Sequence &PopPrograms::streamWeightsFromHostFragment() {
  return seqs[static_cast<int>(ProgramFragmentIndex::STREAMWEIGHTSFROMHOST)];
}

poplar::program::Sequence &PopPrograms::streamOptimizerFromHostFragment() {
  return seqs[static_cast<int>(ProgramFragmentIndex::STREAMOPTIMIZERFROMHOST)];
}

poplar::program::Sequence &PopPrograms::setRandomSeedFragment() {
  return seqs[static_cast<int>(ProgramFragmentIndex::SETRANDOMSEED)];
}

poplar::program::Sequence &PopPrograms::initFragment() {
  return seqs[static_cast<int>(ProgramFragmentIndex::INIT)];
}

poplar::program::Sequence &PopPrograms::preForwardFragment() {
  return seqs[static_cast<int>(ProgramFragmentIndex::PREFORWARD)];
}

poplar::program::Sequence &PopPrograms::forwardFragment() {
  return seqs[static_cast<int>(ProgramFragmentIndex::FORWARD)];
}

poplar::program::Sequence &PopPrograms::backwardFragment() {
  return seqs[static_cast<int>(ProgramFragmentIndex::BACKWARD)];
}

poplar::program::Sequence &PopPrograms::toHostFinalCopyFragment() {
  return seqs[static_cast<int>(ProgramFragmentIndex::TOHOSTFINALCOPY)];
}

poplar::program::Sequence &PopPrograms::varUpdateFromAccumulatorFragment() {
  return seqs[static_cast<int>(ProgramFragmentIndex::VARUPDATEFROMACCUMULATOR)];
}

poplar::program::Sequence &
PopPrograms::resetWeightGradientAccumulatorFragment() {
  return seqs[static_cast<int>(
      ProgramFragmentIndex::RESETWEIGHTGRADIENTACCUMULATOR)];
}

poplar::program::Sequence &PopPrograms::weightsToHostFragment() {
  return seqs[static_cast<int>(ProgramFragmentIndex::WEIGHTSTOHOST)];
}

poplar::program::Sequence PopPrograms::weightsFromHost() {
  poplar::program::Sequence prog;
  prog.add(streamWeightsFromHostFragment());
  return prog;
}

poplar::program::Sequence PopPrograms::optimizerFromHost() {
  poplar::program::Sequence prog;
  prog.add(streamOptimizerFromHostFragment());
  return prog;
}

poplar::program::Sequence PopPrograms::setRandomSeed() {
  poplar::program::Sequence prog;
  prog.add(setRandomSeedFragment());
  return prog;
}

void PopPrograms::addPipelineCycle(PipelineCycle pCycle,
                                   poplar::program::Sequence &sq,
                                   std::ostringstream &ss) {
  // Inside the each phase, conditionally do:
  //
  // 1. The pre-forward fragment
  // 2. Host->Device copies for each IPU
  // 3. Forward fragments for each IPU
  // 4. Increment stash index for each IPU
  // 5. Restore fragment for each IPU
  // 6. Backward fragments for each IPU
  // 7. Device->Host copies for each IPU
  // 8. Inter-IPU copies

  PipelineInfo pInfo = dv_p->pipelineInfo();

  // 1.
  sq.add(preForwardFragment());

  // 2.
  if (pipelineSeqs.find(PipelineFragmentId::ToDeviceStream) !=
      pipelineSeqs.end()) {
    for (auto &vgid_seq : pipelineSeqs.at(PipelineFragmentId::ToDeviceStream)) {
      if (pInfo.doFwd(pCycle, vgid_seq.first)) {
        ss << "\n  vg" << vgid_seq.first << " : ToDeviceStream";
        sq.add(vgid_seq.second);
      }
    }
  } else {
    if (dv_p->useSyntheticData() == false) {
      throw error(
          "There are no ToDeviceStream pipeline program fragments. Check that "
          "the stream copies have been added to the correct fragment.");
    }
  }

  // 3.
  for (auto &vgid_seq : pipelineSeqs.at(PipelineFragmentId::Forward)) {
    if (pInfo.doFwd(pCycle, vgid_seq.first)) {
      ss << "\n  vg" << vgid_seq.first << " : Forward";
      sq.add(vgid_seq.second);
    }
  }

  // 4.
  if (pipelineSeqs.find(PipelineFragmentId::IncrStashIndex) !=
      pipelineSeqs.end()) {
    for (auto &vgid_seq : pipelineSeqs.at(PipelineFragmentId::IncrStashIndex)) {
      if (pInfo.doFwd(pCycle, vgid_seq.first) ||
          pInfo.doBwd(pCycle, vgid_seq.first)) {
        ss << "\n  vg" << vgid_seq.first << " : IncrStashIndex";
        sq.add(vgid_seq.second);
      }
    }
  }

  // 5.
  if (pipelineSeqs.find(PipelineFragmentId::Restore) != pipelineSeqs.end()) {
    for (auto &vgid_seq : pipelineSeqs.at(PipelineFragmentId::Restore)) {
      // of the bwd fragment is added, then the restore fragment must be added
      // too
      if (pInfo.doBwd(pCycle, vgid_seq.first)) {
        ss << "\n  vg" << vgid_seq.first << " : Restore";
        logging::devicex::debug("Adding restore frag to final sq");
        sq.add(vgid_seq.second);
      }
    }
  }

  // 6.
  if (pipelineSeqs.find(PipelineFragmentId::Backward) != pipelineSeqs.end()) {
    for (auto &vgid_seq : pipelineSeqs.at(PipelineFragmentId::Backward)) {
      if (pInfo.doBwd(pCycle, vgid_seq.first)) {
        ss << "\n  vg" << vgid_seq.first << " : Backward";
        sq.add(vgid_seq.second);
      }
    }
  }

  // 7.
  if (pipelineSeqs.find(PipelineFragmentId::FwdToHostStream) !=
      pipelineSeqs.end()) {
    for (auto &vgid_seq :
         pipelineSeqs.at(PipelineFragmentId::FwdToHostStream)) {
      if (pInfo.doFwd(pCycle, vgid_seq.first)) {
        ss << "\n  vg" << vgid_seq.first << " : FwdToHostStream";
        sq.add(vgid_seq.second);
      }
    }
  }
  if (pipelineSeqs.find(PipelineFragmentId::BwdToHostStream) !=
      pipelineSeqs.end()) {
    for (auto &vgid_seq :
         pipelineSeqs.at(PipelineFragmentId::BwdToHostStream)) {
      if (pInfo.doBwd(pCycle, vgid_seq.first)) {
        ss << "\n  vg" << vgid_seq.first << " : BwdToHostStream";
        sq.add(vgid_seq.second);
      }
    }
  }

  // 8.1 Insert the FWD inter IPU-copies.
  // We add these in reverse order, i->i+1 then i-1->i then i-2->i-1 etc.
  // Note that vgid_seq.first == 1 corresponds to the copy from IPU1 to IPU2
  auto foundFwd = pipelineSeqs.find(PipelineFragmentId::IpuCopyFwd);
  if (foundFwd != pipelineSeqs.end()) {
    const auto &M = foundFwd->second;
    for (auto vgid_seq = M.rbegin(); vgid_seq != M.rend(); ++vgid_seq) {
      if (pInfo.doFwd(pCycle, vgid_seq->first)) {
        ss << "\n  vg" << vgid_seq->first << " : IpuFwdCopy";
        sq.add(vgid_seq->second);
      }
    }
  }

  // 8.2 Insert the BWD inter IPU-copies.
  // These are added in ascending order of virtual graph id, so i-1->i-2 then
  // i->i-1 then i+1->i etc. This order is important for correctness.
  // Note that vgid_seq.first == 1 corresponds to the copy from IPU1 to IPU0
  if (pipelineSeqs.find(PipelineFragmentId::IpuCopyBwd) != pipelineSeqs.end()) {
    for (const auto &vgid_seq :
         pipelineSeqs.at(PipelineFragmentId::IpuCopyBwd)) {
      if (pInfo.doBwd(pCycle, vgid_seq.first)) {
        ss << "\n  vg" << vgid_seq.first << " : IpuBwdCopy";
        sq.add(vgid_seq.second);
      }
    }
  }
}

poplar::program::Sequence PopPrograms::getMainProgramFromPipelineFragments() {

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
    for (auto vgid_desc : pipelineDescs.at(fragId)) {
      auto vgStr = std::to_string(vgid_desc.first);
      auto desc  = vgid_desc.second;
      ss << "\n  vg" + vgStr + ":" + desc;
    }
  }
  ss << "\n\n";

  PipelineInfo pInfo = dv_p->pipelineInfo();

  poplar::program::Sequence fill;
  for (PipelineCycle pCycle = pInfo.fillPhase.start;
       pCycle <= pInfo.fillPhase.end;
       pCycle++) {
    ss << "\nPipeline Cycle " + std::to_string(pCycle) + ":";
    addPipelineCycle(pCycle, fill, ss);
  }

  // All pipeline cycles in the main phase are identical. So we create the
  // program for a single cycle and repeat for mainCycles
  poplar::program::Sequence main;
  int64_t mainCycles = pInfo.mainPhase.end - pInfo.mainPhase.start + 1;
  ss << "\nPipeline Cycle 'Main', " + std::to_string(mainCycles) + " cycles";
  addPipelineCycle(pInfo.mainPhase.start, main, ss);

  poplar::program::Sequence flush;
  for (PipelineCycle pCycle = pInfo.flushPhase.start;
       pCycle <= pInfo.flushPhase.end;
       pCycle++) {
    ss << "\nPipeline Cycle " + std::to_string(pCycle) + ":";
    addPipelineCycle(pCycle, flush, ss);
  }

  logging::devicex::debug("Pipelining program construction summary:");
  logging::devicex::debug(ss.str());

  poplar::program::Sequence inner;

  inner.add(initFragment());
  inner.add(fill);
  // This is the inner main cycles loop, if doing pipelining withour gradient
  // accumulation, this the batches per step loop, as batch size = micro_batch
  // size
  inner.add(poplar::program::Repeat(static_cast<int>(mainCycles), main));
  inner.add(flush);
  poplar::program::Sequence outer;

  if (dv_p->ir().getSessionOptions().enableGradientAccumulation) {

    inner.add(varUpdateFromAccumulatorFragment());
    inner.add(resetWeightGradientAccumulatorFragment());
    // If doing gradient accumulation, the inner loop loops over mini batches,
    // and this outer loop loops over multiple batches per step.
    outer = poplar::program::Repeat(
        static_cast<int>(dv_p->ir().getDataFlow().batchesPerStep()), inner);
  } else {
    // No gradient accumulation, so just add one iteration of the inner program.
    outer.add(inner);
  }

  outer.add(toHostFinalCopyFragment());

  return outer;
}

poplar::program::Sequence PopPrograms::program() {
  if (dv_p->ir().getSessionOptions().enablePipelining) {
    return getMainProgramFromPipelineFragments();
  } else {
    poplar::program::Sequence prog;
    prog.add(preForwardFragment());
    prog.add(forwardFragment());
    prog.add(backwardFragment());

    poplar::program::Sequence outer;

    outer.add(initFragment());

    auto accumulationFactor = static_cast<int>(dv_p->getAccumulationFactor());
    if (dv_p->ir().getSessionOptions().enableGradientAccumulation) {
      logging::devicex::trace(
          "Adding gradient accumulation repeat loop with {} loops",
          accumulationFactor);
      prog = poplar::program::Repeat(accumulationFactor, prog);
      prog.add(varUpdateFromAccumulatorFragment());
      prog.add(resetWeightGradientAccumulatorFragment());
    }

    // BatchesPerStep loop
    outer.add(poplar::program::Repeat(dv_p->ir().getDataFlow().batchesPerStep(),
                                      prog));
    outer.add(toHostFinalCopyFragment());

    return outer;
  }
}

poplar::program::Sequence PopPrograms::weightsToHost() {
  return weightsToHostFragment();
}

std::vector<poplar::program::Program> PopPrograms::progs() {
  std::vector<poplar::program::Program> ps(ProgramIndex::N);

  ps[ProgramIndex::WEIGHTSFROMHOST]   = weightsFromHost();
  ps[ProgramIndex::OPTIMIZERFROMHOST] = optimizerFromHost();
  ps[ProgramIndex::SETRANDOMSEED]     = setRandomSeed();
  ps[ProgramIndex::PROGRAM]           = program();
  ps[ProgramIndex::WEIGHTSTOHOST]     = weightsToHost();

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

bool PopPrograms::hasBeenRecomputed(OpId id) const {
  auto itHas = (beenRecomputed.find(id) != beenRecomputed.end());
  return itHas;
}

void PopPrograms::recordRecomputed(OpId id) { beenRecomputed.insert(id); }

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
PopPrograms::pipelineFragment(VGraphId vGraphId,
                              PipelineFragmentId frag,
                              const std::string &desc) {
  auto foundFrag = pipelineSeqs.find(frag);
  if (foundFrag != pipelineSeqs.end()) {
    auto foundVGraph = pipelineSeqs.at(frag).find(vGraphId);
    if (foundVGraph != pipelineSeqs.at(frag).end()) {
      pipelineDescs.at(frag).at(vGraphId).append("\n    " + desc);
      return pipelineSeqs.at(frag).at(vGraphId);
    } else {
      pipelineDescs.at(frag).insert({vGraphId, "\n    " + desc});
      pipelineSeqs.at(frag).insert({vGraphId, poplar::program::Sequence{}});
      return pipelineSeqs.at(frag).at(vGraphId);
    }
  } else {
    pipelineDescs.insert({frag, {{vGraphId, "\n    " + desc}}});
    pipelineSeqs.insert({frag, {{vGraphId, poplar::program::Sequence{}}}});
    return pipelineSeqs.at(frag).at(vGraphId);
  }
}

poplar::program::Sequence &
PopPrograms::pipelineForwardFragment(VGraphId vGraphId,
                                     const std::string &desc) {
  return pipelineFragment(vGraphId, PipelineFragmentId::Forward, desc);
}

poplar::program::Sequence &
PopPrograms::pipelineBackwardFragment(VGraphId vGraphId,
                                      const std::string &desc) {
  return pipelineFragment(vGraphId, PipelineFragmentId::Backward, desc);
}

poplar::program::Sequence &
PopPrograms::pipelineToDeviceStreamFragment(VGraphId vGraphId,
                                            const std::string &desc) {
  return pipelineFragment(vGraphId, PipelineFragmentId::ToDeviceStream, desc);
}

poplar::program::Sequence &
PopPrograms::pipelineFwdToHostStreamFragment(VGraphId vGraphId,
                                             const std::string &desc) {
  return pipelineFragment(vGraphId, PipelineFragmentId::FwdToHostStream, desc);
}

poplar::program::Sequence &
PopPrograms::pipelineRestoreFragment(VGraphId vGraphId,
                                     const std::string &desc) {
  return pipelineFragment(vGraphId, PipelineFragmentId::Restore, desc);
}

poplar::program::Sequence &
PopPrograms::pipelineBwdToHostStreamFragment(VGraphId vGraphId,
                                             const std::string &desc) {
  return pipelineFragment(vGraphId, PipelineFragmentId::BwdToHostStream, desc);
}

poplar::program::Sequence &
PopPrograms::pipelineIncrStashIndexFragment(VGraphId vGraphId,
                                            const std::string &desc) {
  return pipelineFragment(vGraphId, PipelineFragmentId::IncrStashIndex, desc);
}

poplar::program::Sequence &
PopPrograms::pipelineIpuCopyFwdFragment(VGraphId id, const std::string &desc) {
  return pipelineFragment(id, PipelineFragmentId::IpuCopyFwd, desc);
}

poplar::program::Sequence &
PopPrograms::pipelineIpuCopyBwdFragment(VGraphId id, const std::string &desc) {
  return pipelineFragment(id, PipelineFragmentId::IpuCopyBwd, desc);
}

poplar::program::Sequence &
PopPrograms::pipelineFwdOrBwdToHostStreamFragment(ScheduledPreLoss preLoss,
                                                  VGraphId vGraphId,
                                                  const std::string &desc) {
  switch (preLoss) {
  case ScheduledPreLoss::Yes: {
    return pipelineFwdToHostStreamFragment(vGraphId, desc);
  }
  case ScheduledPreLoss::No: {
    return pipelineBwdToHostStreamFragment(vGraphId, desc);
  }
  case ScheduledPreLoss::Undefined: {
    throw error("There is no fragment for Undefined SchedulePreLoss");
  }
  }
}

std::string
PopPrograms::getStrFromPipelineFragmentId(PipelineFragmentId fragId) {
  switch (fragId) {
  case PipelineFragmentId::ToDeviceStream: {
    return "ToDeviceStream";
  }
  case PipelineFragmentId::Forward: {
    return "Forward";
  }
  case PipelineFragmentId::Backward: {
    return "Backward";
  }
  case PipelineFragmentId::FwdToHostStream: {
    return "FwdToHostStream";
  }
  case PipelineFragmentId::BwdToHostStream: {
    return "BwdToHostStream";
  }
  case PipelineFragmentId::IncrStashIndex: {
    return "IncrStashIndex";
  }
  case PipelineFragmentId::IpuCopyFwd: {
    return "IpuCopyFwd";
  }
  case PipelineFragmentId::IpuCopyBwd: {
    return "IpuCopyBwd";
  }

  case PipelineFragmentId::Restore: {
    return "Restore";
  }

  case PipelineFragmentId::N: {
    throw error("Cannot return string for PipelineFragmentId 'N'");
  }
  }
}

} // namespace popx
} // namespace popart
