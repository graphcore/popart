// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <popef/Types.hpp>
#include <poplar/FunctionBufferMappingType.hpp>
#include <poplar/Graph.hpp>
#include <poplar/GraphElements.hpp>
#include <popops/Zero.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/popx/irlowering.hpp>
#include <popart/popx/popprograms.hpp>

#include "popart/dataflow.hpp"
#include "popart/devicemanager.hpp"
#include "popart/error.hpp"
#include "popart/graphid.hpp"
#include "popart/logging.hpp"
#include "popart/names.hpp"
#include "popart/popx/debugcontextx.hpp"
#include "popart/popx/pritask.hpp"
#include "popart/sessionoptions.hpp"
#include "popart/transforms/pipeline.hpp"
#include "popart/vertex.hpp"

namespace popart {
namespace popx {

std::ostream &operator<<(std::ostream &out, PopPrograms::ProgramIndex);

const std::unordered_map<popef::ProgramFlow::ProgramIndexType, std::string>
    PopPrograms::commonPrograms = []() {
      std::unordered_map<popef::ProgramFlow::ProgramIndexType, std::string> out;

      for (int i = 0; i < PopPrograms::ProgramIndex::N; i++) {
        std::stringstream ss;
        ss << static_cast<PopPrograms::ProgramIndex>(i);
        out[i] = ss.str();
      }

      return out;
    }();

std::ostream &operator<<(std::ostream &out, PopPrograms::ProgramIndex index) {
  switch (index) {
  case PopPrograms::ProgramIndex::WeightsFromHost:
    out << "WeightsFromHost";
    break;
  case PopPrograms::ProgramIndex::OptimizerFromHost:
    out << "OptimizerFromHost";
    break;
  case PopPrograms::ProgramIndex::RandomSeedFromHost:
    out << "RandomSeedFromHost";
    break;
  case PopPrograms::ProgramIndex::RandomSeedToHost:
    out << "RandomSeedToHost";
    break;
  case PopPrograms::ProgramIndex::RngStateFromHost:
    out << "RngStateFromHost";
    break;
  case PopPrograms::ProgramIndex::Program:
    out << "Program";
    break;
  case PopPrograms::ProgramIndex::RngStateToHost:
    out << "RngStateToHost";
    break;
  case PopPrograms::ProgramIndex::WeightsToHost:
    out << "WeightsToHost";
    break;
  case PopPrograms::ProgramIndex::CycleCountTensorToHost:
    out << "CycleCountTensorToHost";
    break;
  case PopPrograms::ProgramIndex::CustomProgramsStart:
    out << "CustomProgramsStart";
    break;
  case PopPrograms::ProgramIndex::N:
    out << "N";
    break;
  default: {
    throw internal_error("Invalid value for ProgramIndex");
  }
  };

  return out;
}

std::ostream &operator<<(std::ostream &out,
                         PopPrograms::ProgramFragmentIndex index) {
  switch (index) {
  case PopPrograms::ProgramFragmentIndex::StreamWeightsFromHost: {
    out << "StreamWeightsFromHost";
    break;
  }
  case PopPrograms::ProgramFragmentIndex::StreamOptimizerFromHost: {
    out << "StreamOptimizerFromHost";
    break;
  }
  case PopPrograms::ProgramFragmentIndex::RandomSeedFromHost: {
    out << "RandomSeedFromHost";
    break;
  }
  case PopPrograms::ProgramFragmentIndex::RandomSeedToHost: {
    out << "RandomSeedToHost";
    break;
  }
  case PopPrograms::ProgramFragmentIndex::RngStateFromHost: {
    out << "RngStateFromHost";
    break;
  }
  case PopPrograms::ProgramFragmentIndex::Init: {
    out << "Init";
    break;
  }
  case PopPrograms::ProgramFragmentIndex::PreForward: {
    out << "PreForward";
    break;
  }
  case PopPrograms::ProgramFragmentIndex::Forward: {
    out << "Forward";
    break;
  }
  case PopPrograms::ProgramFragmentIndex::Backward: {
    out << "Backward";
    break;
  }
  case PopPrograms::ProgramFragmentIndex::VarUpdateFromAccumulator: {
    out << "VarUpdateFromAccumulator";
    break;
  }
  case PopPrograms::ProgramFragmentIndex::RngStateToHost: {
    out << "RngStateToHost";
    break;
  }
  case PopPrograms::ProgramFragmentIndex::WeightsToHost: {
    out << "WeightsToHost";
    break;
  }
  case PopPrograms::ProgramFragmentIndex::ToHostFinalCopy: {
    out << "ToHostFinalCopy";
    break;
  }
  case PopPrograms::ProgramFragmentIndex::CycleCountTensorToHost: {
    out << "CycleCountTensorToHost";
    break;
  }
  case PopPrograms::ProgramFragmentIndex::N: {
    out << "N";
    break;
  }
  default: {
    throw internal_error("Invalid value for ProgramFragmentIndex");
  }
  };
  return out;
}

PopPrograms::PopPrograms(IrLowering *ir_lowering_p_)
    : ir_lowering_p(ir_lowering_p_) {
  // Populate seqs with Sequences that have names.
  for (int i = 0; i < static_cast<int>(ProgramFragmentIndex::N); ++i) {
    std::stringstream ss;
    ss << static_cast<ProgramFragmentIndex>(i);
    seqs.push_back(poplar::program::Sequence({}, ss.str()));
  }
}

const poplar::program::Sequence &
PopPrograms::streamWeightsFromHostFragment() const {
  return seqs.at(static_cast<int>(ProgramFragmentIndex::StreamWeightsFromHost));
}
poplar::program::Sequence &PopPrograms::streamWeightsFromHostFragment() {
  return seqs.at(static_cast<int>(ProgramFragmentIndex::StreamWeightsFromHost));
}

const poplar::program::Sequence &
PopPrograms::streamOptimizerFromHostFragment() const {
  return seqs.at(
      static_cast<int>(ProgramFragmentIndex::StreamOptimizerFromHost));
}
poplar::program::Sequence &PopPrograms::streamOptimizerFromHostFragment() {
  return seqs.at(
      static_cast<int>(ProgramFragmentIndex::StreamOptimizerFromHost));
}

const poplar::program::Sequence &
PopPrograms::randomSeedFromHostFragment() const {
  return seqs.at(static_cast<int>(ProgramFragmentIndex::RandomSeedFromHost));
}
poplar::program::Sequence &PopPrograms::randomSeedFromHostFragment() {
  return seqs.at(static_cast<int>(ProgramFragmentIndex::RandomSeedFromHost));
}
const poplar::program::Sequence &PopPrograms::randomSeedToHostFragment() const {
  return seqs.at(static_cast<int>(ProgramFragmentIndex::RandomSeedToHost));
}
poplar::program::Sequence &PopPrograms::randomSeedToHostFragment() {
  return seqs.at(static_cast<int>(ProgramFragmentIndex::RandomSeedToHost));
}

const poplar::program::Sequence &PopPrograms::rngStateFromHostFragment() const {
  return seqs.at(static_cast<int>(ProgramFragmentIndex::RngStateFromHost));
}

poplar::program::Sequence &PopPrograms::rngStateFromHostFragment() {
  return seqs.at(static_cast<int>(ProgramFragmentIndex::RngStateFromHost));
}

const poplar::program::Sequence &PopPrograms::rngStateToHostFragment() const {
  return seqs.at(static_cast<int>(ProgramFragmentIndex::RngStateToHost));
}

poplar::program::Sequence &PopPrograms::rngStateToHostFragment() {
  return seqs.at(static_cast<int>(ProgramFragmentIndex::RngStateToHost));
}

const poplar::program::Sequence &
PopPrograms::cycleCountTensorToHostFragment() const {
  return seqs.at(
      static_cast<int>(ProgramFragmentIndex::CycleCountTensorToHost));
}
poplar::program::Sequence &PopPrograms::cycleCountTensorToHostFragment() {
  return seqs.at(
      static_cast<int>(ProgramFragmentIndex::CycleCountTensorToHost));
}

const poplar::program::Sequence &PopPrograms::initFragment() const {
  return seqs.at(static_cast<int>(ProgramFragmentIndex::Init));
}

poplar::program::Sequence &PopPrograms::initFragment() {
  return seqs.at(static_cast<int>(ProgramFragmentIndex::Init));
}

const poplar::program::Sequence &PopPrograms::preForwardFragment() const {
  return seqs.at(static_cast<int>(ProgramFragmentIndex::PreForward));
}

poplar::program::Sequence &PopPrograms::preForwardFragment() {
  return seqs.at(static_cast<int>(ProgramFragmentIndex::PreForward));
}

const poplar::program::Sequence &PopPrograms::forwardFragment() const {
  return seqs.at(static_cast<int>(ProgramFragmentIndex::Forward));
}

poplar::program::Sequence &PopPrograms::forwardFragment() {
  return seqs.at(static_cast<int>(ProgramFragmentIndex::Forward));
}

const poplar::program::Sequence &PopPrograms::backwardFragment() const {
  return seqs.at(static_cast<int>(ProgramFragmentIndex::Backward));
}

poplar::program::Sequence &PopPrograms::backwardFragment() {
  return seqs.at(static_cast<int>(ProgramFragmentIndex::Backward));
}

const poplar::program::Sequence &PopPrograms::toHostFinalCopyFragment() const {
  return seqs.at(static_cast<int>(ProgramFragmentIndex::ToHostFinalCopy));
}

poplar::program::Sequence &PopPrograms::toHostFinalCopyFragment() {
  return seqs.at(static_cast<int>(ProgramFragmentIndex::ToHostFinalCopy));
}

const poplar::program::Sequence &PopPrograms::accumulateOuterFragment() const {
  return seqs.at(
      static_cast<int>(ProgramFragmentIndex::VarUpdateFromAccumulator));
}

poplar::program::Sequence &PopPrograms::accumulateOuterFragment() {
  return seqs.at(
      static_cast<int>(ProgramFragmentIndex::VarUpdateFromAccumulator));
}

const poplar::program::Sequence &PopPrograms::weightsToHostFragment() const {
  return seqs.at(static_cast<int>(ProgramFragmentIndex::WeightsToHost));
}

poplar::program::Sequence &PopPrograms::weightsToHostFragment() {
  return seqs.at(static_cast<int>(ProgramFragmentIndex::WeightsToHost));
}

poplar::program::Sequence PopPrograms::weightsFromHost() const {
  poplar::program::Sequence prog(poplar::DebugContext{"weightsFromHost"});
  prog.add(streamWeightsFromHostFragment());
  return prog;
}

poplar::program::Sequence PopPrograms::optimizerFromHost() const {
  poplar::program::Sequence prog(poplar::DebugContext{"optimizerFromHost"});
  prog.add(streamOptimizerFromHostFragment());
  return prog;
}

poplar::program::Sequence PopPrograms::randomSeedFromHost() const {
  poplar::program::Sequence prog(poplar::DebugContext{"randomSeedFromHost"});
  prog.add(randomSeedFromHostFragment());
  return prog;
}

poplar::program::Sequence PopPrograms::randomSeedToHost() const {
  poplar::program::Sequence prog(poplar::DebugContext{"randomSeedToHost"});
  prog.add(randomSeedToHostFragment());
  return prog;
}

poplar::program::Sequence PopPrograms::cycleCountTensorToHost() const {
  poplar::program::Sequence prog(
      poplar::DebugContext{"cycleCountTensorToHost"});
  prog.add(cycleCountTensorToHostFragment());
  return prog;
}

poplar::program::Sequence PopPrograms::rngStateFromHost() const {
  poplar::program::Sequence prog(poplar::DebugContext{"rngStateFromHost"});
  prog.add(rngStateFromHostFragment());
  return prog;
}

poplar::program::Sequence PopPrograms::rngStateToHost() const {
  poplar::program::Sequence prog(poplar::DebugContext{"rngStateToHost"});
  prog.add(rngStateToHostFragment());
  return prog;
}

void PopPrograms::addPipelineCycle(PipelineInfo pInfo,
                                   PipelineCycle pCycle,
                                   poplar::program::Sequence &sq,
                                   std::ostringstream &ss) const {
  // Inside each pipeline cycle
  //
  // Always do:
  // 1. The pre-forward fragment
  //
  // Then conditionally do:
  // 2. Host->Device copies for each pipeline stage
  // 3. Main fragments for each pipeline stage
  // 4. Device->Host copies for each pipeline stage
  //
  // Then always do:
  // 5. Inter-IPU copies for all pipeline stages

  // 1.
  sq.add(preForwardFragment());

  // 2.
  if (pipelineSeqs.find(PipelineFragmentId::ToDeviceStream) !=
      pipelineSeqs.end()) {
    for (const auto &stage_seq :
         pipelineSeqs.at(PipelineFragmentId::ToDeviceStream)) {
      if (pInfo.doStage(pCycle, stage_seq.first)) {
        ss << "\n  ps" << stage_seq.first << " : ToDeviceStream";
        // Inline code in order to not block overlap
        sq.add(stage_seq.second);
      }
    }
  } else {
    if (ir_lowering_p->ir().useSyntheticData() == false) {
      throw error(
          "There are no ToDeviceStream pipeline program fragments. Check that "
          "the stream copies have been added to the correct fragment.");
    }
  }

  // 3.
  for (const auto &stage_seq : mainPipelineFunctions) {
    auto stage = stage_seq.first;
    if (pInfo.doStage(pCycle, stage)) {
      ss << "\n  ps" << stage << " : Main";
      sq.add(poplar::program::Call(stage_seq.second));
    }
  }

  // 4.
  if (pipelineSeqs.find(PipelineFragmentId::ToHostStream) !=
      pipelineSeqs.end()) {
    for (const auto &stage_seq :
         pipelineSeqs.at(PipelineFragmentId::ToHostStream)) {
      if (pInfo.doStage(pCycle, stage_seq.first)) {
        ss << "\n  ps" << stage_seq.first << " : ToHostStream";
        // Inline code to not block overlap
        sq.add(stage_seq.second);
      }
    }
  }

  // 5.
  // Note: Always do all the copies. This is ensure that ALL copies are
  // outlined across pipelineCycles AND merged across pipelineStages.
  ss << logging::format("\n  IpuCopies");
  sq.add(poplar::program::Call(pipelineIpuCopyFunction));
}

void PopPrograms::addFunctionBuffers(const GraphId gid,
                                     poplar::FunctionBufferMappingType fbmt) {
  auto &g = ir_lowering_p->ir().getGraph(gid);

  if (hasFunctionBuffer(gid, fbmt)) {
    // Do nothing - a previous code load op has already created a buffer for
    // this function.
  } else {
    FunctionBuffers func_vec;
    auto &graph_progs = ir_lowering_p->getFragmentFunctions(g);

    for (auto f : graph_progs) {
      auto buffer = ir_lowering_p->graph().addFunctionBuffer(f, fbmt);
      func_vec.push_back({f, buffer});
    }
    std::pair<const GraphId, poplar::FunctionBufferMappingType> pair = {gid,
                                                                        fbmt};
    functionBuffers.emplace(pair, func_vec);
  }
}

unsigned
PopPrograms::addCustomProgram(const poplar::program::Program &program) {
  customPrograms.push_back(program);
  return ProgramIndex::CustomProgramsStart + customPrograms.size() - 1;
}

void PopPrograms::createPipelineFunctions() {
  // Function to reset all stash and restore indices when switching between
  // training and inference in the same engine. This is required because
  // the inference program will have fewer pipeline stages (forward stages only)
  // and therefore fewer pipeline stash entries. This will mean that after
  // training, all indices will be at 0 again, however, after an inference step,
  // not all indices are 0 and need to be reset to make the next inference
  // or training step correct.
  {
    poplar::program::Sequence sequence;
    for (auto &index : ir_lowering_p->getPipelineIndexTensors()) {
      popops::zero(
          ir_lowering_p->graph(), index, sequence, {"zeroPipelineIndex"});
    }
    zeroPipelineIndexFunction = ir_lowering_p->graph().addFunction(sequence);
  }

  for (const auto &stage_seq : pipelineSeqs.at(PipelineFragmentId::Main)) {
    // fwdOnly: Skip stages containing backward pass only
    const poplar::program::Sequence &sequence = stage_seq.second;
    mainPipelineFunctions.insert(
        {stage_seq.first, ir_lowering_p->graph().addFunction(sequence)});
  }

  pipelineIpuCopyFunction =
      ir_lowering_p->graph().addFunction(pipelineIpuCopySeq);
  toHostFinalCopyFunction =
      ir_lowering_p->graph().addFunction(toHostFinalCopyFragment());
}

poplar::program::Sequence
PopPrograms::getFullProgramFromPipelineFragments(bool fwdOnly) const {
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

  PipelineInfo pInfo = ir_lowering_p->ir().pipelineInfo();

  // Adjust pipeline info
  if (fwdOnly) {
    logging::info("[PopPrograms::getFullProgramFromPipelineFragments] Creating "
                  "forward-only program with pipeline stages [{},{}]",
                  0,
                  ir_lowering_p->ir().getNumPipelineStages() / 2 - 1);

    // Skip stages containing backward pass only
    pInfo =
        PipelineInfo(static_cast<int64_t>(
                         ir_lowering_p->ir().getDataFlow().batchesPerStep()),
                     ir_lowering_p->ir().getSessionOptions().accumulationFactor,
                     ir_lowering_p->ir().getNumPipelineStages() / 2,
                     pInfo.doGradAccl,
                     pInfo.withStage);
  }

  poplar::program::Sequence fill(poplar::DebugContext{"fill"});
  for (PipelineCycle pCycle = pInfo.fillPhase.start;
       pCycle <= pInfo.fillPhase.end;
       pCycle++) {
    ss << "\nPipeline Cycle " + std::to_string(pCycle) + ":";
    addPipelineCycle(pInfo, pCycle, fill, ss);
  }

  // All pipeline cycles in the main phase are identical. So we create the
  // program for a single cycle and repeat for mainCycles
  poplar::program::Sequence main(poplar::DebugContext{"main"});
  int64_t mainCycles = pInfo.getMainCycles();
  ss << "\nPipeline Cycle 'Main', " + std::to_string(mainCycles) + " cycles";
  addPipelineCycle(pInfo, pInfo.mainPhase.start, main, ss);

  poplar::program::Sequence flush(poplar::DebugContext{"flush"});
  for (PipelineCycle pCycle = pInfo.flushPhase.start;
       pCycle <= pInfo.flushPhase.end;
       pCycle++) {
    ss << "\nPipeline Cycle " + std::to_string(pCycle) + ":";
    addPipelineCycle(pInfo, pCycle, flush, ss);
  }

  logging::devicex::debug("Pipelining program construction summary:");
  logging::devicex::debug(ss.str());

  poplar::program::Sequence inner(poplar::DebugContext{"inner"});

  inner.add(fill);
  // This is the inner main cycles loop, if doing pipelining without gradient
  // accumulation, this the batches per step loop, as batch size = micro_batch
  // size
  inner.add(poplar::program::Repeat(
      static_cast<uint32_t>(mainCycles), main, {"innerLoop"}));
  inner.add(flush);

  poplar::program::Sequence outer(poplar::DebugContext{"outer"});

  outer.add(initFragment());

  // Only add index zero function call if required
  if (ir_lowering_p->ir()
          .getSessionOptions()
          .createImplicitPipeliningFwdOnlyProgram) {
    outer.add(poplar::program::Call(zeroPipelineIndexFunction));
  }

  if (!ir_lowering_p->getOuterLoopFragEmpty()) {
    if (!fwdOnly) {
      inner.add(accumulateOuterFragment());
    }
    // If doing gradient accumulation, the inner loop is over mini batches,
    // and this outer loop loops over multiple batches per step.
    auto bps = ir_lowering_p->ir().getDataFlow().batchesPerStep();
    outer.add(poplar::program::Repeat(bps, inner, {"outerloop"}));
  } else {
    // No gradient accumulation, so just add one iteration of the inner program.
    outer.add(inner);
  }

  outer.add(poplar::program::Call(toHostFinalCopyFunction));

  return outer;
}

poplar::program::Sequence PopPrograms::program() const {
  const auto &opts      = ir_lowering_p->ir().getSessionOptions();
  auto instrumentations = opts.hardwareInstrumentations;

  poplar::program::Sequence outer(poplar::DebugContext{"outer"});

  if (opts.enableExplicitMainLoops) {
    outer.add(initFragment());
    outer.add(preForwardFragment());
    outer.add(forwardFragment());
    outer.add(backwardFragment());
    outer.add(toHostFinalCopyFragment());
  } else {
    if (opts.implicitPipeliningEnabled()) {
      outer.add(getFullProgramFromPipelineFragments(false));
    } else {
      poplar::program::Sequence prog(poplar::DebugContext{"program"});
      prog.add(preForwardFragment());
      prog.add(forwardFragment());
      prog.add(backwardFragment());

      outer.add(initFragment());

      auto accumulationFactor = ir_lowering_p->getAccumulationFactor();
      if (!ir_lowering_p->getOuterLoopFragEmpty()) {
        logging::devicex::trace(
            "Adding gradient accumulation repeat loop with {} iterations",
            accumulationFactor);
        poplar::program::Repeat repeat(
            accumulationFactor, prog, {"accumulationLoop"});
        prog = poplar::program::Sequence{};
        prog.add(repeat);
        prog.add(accumulateOuterFragment());
      }

      if (opts.instrumentWithHardwareCycleCounter &&
          instrumentations.find(Instrumentation::Inner) !=
              instrumentations.end()) {
        // Instrument first tile of every IPU for inner program
        auto numIpus = ir_lowering_p->getDeviceInfo()->getNumIpus() /
                       ir_lowering_p->getReplicationFactor();
        for (int64_t i = 0; i < numIpus; ++i) {
          std::stringstream ss;
          // String to identify instrumentation
          ss << "inner_ipu_" << i;
          ir_lowering_p->instrumentWithHardwareCycleCounter(
              prog,
              i * static_cast<int64_t>(
                      ir_lowering_p->getDeviceInfo()->getTilesPerIPU()),
              ss.str());
        }
      }

      auto batchesPerStep = ir_lowering_p->ir().getDataFlow().batchesPerStep();
      // BatchesPerStep loop
      logging::devicex::trace("Adding batches per step loop with {} iterations",
                              batchesPerStep);
      outer.add(
          poplar::program::Repeat(batchesPerStep, prog, {"batchesPerStep"}));
      outer.add(toHostFinalCopyFragment());
    }
  }

  if (opts.instrumentWithHardwareCycleCounter &&
      instrumentations.find(Instrumentation::Outer) != instrumentations.end()) {
    ir_lowering_p->instrumentWithHardwareCycleCounter(outer);
  }

  return outer;
}

poplar::program::Sequence PopPrograms::weightsToHost() const {
  poplar::program::Sequence prog(poplar::DebugContext{"weightsToHost"});
  prog.add(weightsToHostFragment());
  return prog;
}

const std::vector<poplar::program::Program> PopPrograms::progs() const {
  std::vector<poplar::program::Program> ps(ProgramIndex::CustomProgramsStart +
                                           customPrograms.size());

  ps[ProgramIndex::WeightsFromHost]        = weightsFromHost();
  ps[ProgramIndex::OptimizerFromHost]      = optimizerFromHost();
  ps[ProgramIndex::RandomSeedFromHost]     = randomSeedFromHost();
  ps[ProgramIndex::RandomSeedToHost]       = randomSeedToHost();
  ps[ProgramIndex::RngStateFromHost]       = rngStateFromHost();
  ps[ProgramIndex::Program]                = program();
  ps[ProgramIndex::RngStateToHost]         = rngStateToHost();
  ps[ProgramIndex::WeightsToHost]          = weightsToHost();
  ps[ProgramIndex::CycleCountTensorToHost] = cycleCountTensorToHost();

  // Add custom programs
  size_t i = 0;
  for (const auto &customProgram : customPrograms) {
    ps[ProgramIndex::CustomProgramsStart + i] = customProgram;
    ++i;
  }

  return ps;
}

poplar::program::Sequence &
PopPrograms::programFragment(PopPrograms::ProgramFragmentIndex index) {
  return seqs.at(static_cast<int>(index));
}

int PopPrograms::getNumFragments(const Graph &graph) const {
  auto scopeIt = scopeSeqs.find(graph.id.str());
  if (scopeIt == scopeSeqs.end()) {
    throw error("There are no scope fragments for {}", graph.getGraphString());
  } else {
    return scopeIt->second.size();
  }
}

std::vector<poplar::program::Sequence> &
PopPrograms::scopeFragments(const Graph &graph) {
  auto scopeIt = scopeSeqs.find(graph.id.str());
  if (scopeIt == scopeSeqs.end()) {
    throw error("There are no scope fragments for {}", graph.getGraphString());
  } else {
    return scopeIt->second;
  }
}

poplar::program::Sequence &
PopPrograms::scopeFragment(const Graph &graph, SubgraphPartIndex subgraphPart) {
  return scopeSeqs.at(graph.id.str()).at(subgraphPart);
}

bool PopPrograms::containsFragments(const Graph &graph) const {
  auto scopeIt = scopeSeqs.find(graph.id.str());
  return (scopeIt != scopeSeqs.end());
}

bool PopPrograms::containsFragment(const Graph &graph,
                                   SubgraphPartIndex subgraphPart) const {

  auto scopeIt = scopeSeqs.find(graph.id.str());
  return (scopeIt != scopeSeqs.end()) &&
         (subgraphPart < scopeIt->second.size());
}

void PopPrograms::createFragment(const Graph &graph,
                                 SubgraphPartIndex subgraphPart) {

  // We only populate scopeSeqs here, funcs needs to be populated after
  // sequences are grown because sequences are cloned on addFunction.
  auto scopeIt = scopeSeqs.find(graph.id.str());

  // Check if this graph has a vector already.
  if (scopeIt != scopeSeqs.end()) {
    // Vectors exists, check it contains the subgraphPart.
    auto &seqs = scopeIt->second;
    if (seqs.size() < subgraphPart + 1) {
      // Check funcs matches scopeSeqs.
      assert(funcs.size() < subgraphPart + 1);
      // Resize scopeSeqs. The funcs vector will be resized to match below.
      seqs.resize(subgraphPart + 1);
    }
  } else {
    // Vector does not exist, create one that contains the subgraph part.
    std::vector<poplar::program::Sequence> seqs;

    for (size_t part = 0; part <= subgraphPart; ++part) {
      std::stringstream dbgCtx;
      if (graph.id.str() == "") {
        dbgCtx << "main_graph/" << part;
      } else {
        dbgCtx << graph.id.str() << "/" << part;
      }
      seqs.push_back(poplar::program::Sequence(dbgCtx.str()));
    }

    scopeSeqs.insert({graph.id.str(), seqs});
  }
}

std::vector<poplar::Function> &
PopPrograms::getFragmentFunctions(const Graph &graph,
                                  poplar::Graph &poplarGraph) {

  auto seq2func = [&](poplar::program::Sequence &seq) {
    return poplarGraph.addFunction(seq);
  };

  auto funcsIt = funcs.find(graph.id.str());
  if (funcsIt == funcs.end()) {
    // Funcs was never populated. Populate it now.
    funcsIt     = funcs.insert({graph.id.str(), {}}).first;
    auto &funcs = funcsIt->second;
    auto &seqs  = scopeSeqs.at(graph.id.str());
    std::transform(
        seqs.begin(), seqs.end(), std::back_inserter(funcs), seq2func);
  }

  if (funcsIt == funcs.end()) {
    throw error("There are no scope fragments for {}", graph.getGraphString());
  }
  return funcsIt->second;
}

poplar::Function &
PopPrograms::getFragmentFunction(const Graph &graph,
                                 SubgraphPartIndex subgraphPart,
                                 poplar::Graph &poplarGraph) {

  auto &funcs = getFragmentFunctions(graph, poplarGraph);

  if (subgraphPart >= funcs.size()) {
    throw error("There is no scope fragment for {}, part {}",
                graph.getGraphString(),
                subgraphPart);
  } else {
    return funcs[subgraphPart];
  }
}

bool PopPrograms::hasBeenRecomputed(OpId id, ExecutionPhase phase) const {
  auto itHas = (beenRecomputed.find({id, phase}) != beenRecomputed.end());
  return itHas;
}

void PopPrograms::recordRecomputed(OpId id, ExecutionPhase phase) {
  beenRecomputed.insert({id, phase});
}

std::vector<poplar::program::Sequence>::iterator
PopPrograms::recomputeFragment(OpId id) {
  auto found = recomputeSeqs.find(id);
  if (found == recomputeSeqs.end()) {
    throw error("Recompute Fragment for Op {} has not been created.", id);
  }
  return found->second.begin();
}

SequenceMap::SequenceInterval PopPrograms::createRecomputeFragment(OpId id) {
  recomputeSeqs.insert({id, {poplar::program::Sequence{}}});
  return SequenceMap::SequenceInterval(recomputeSeqs[id].begin(),
                                       recomputeSeqs[id].end());
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
PopPrograms::pipelineMainFragment(PipelineStage pipelineStage,
                                  const std::string &desc) {
  return pipelineFragment(pipelineStage, PipelineFragmentId::Main, desc);
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
  case PipelineFragmentId::Main: {
    return "Main";
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
