// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_LIVENESS_HPP
#define GUARD_NEURALNET_LIVENESS_HPP

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include <popart/graphid.hpp>
#include <popart/op.hpp>
#include <popart/tensor.hpp>

#include "popart/names.hpp"
#include "popart/tensordebuginfo.hpp"

namespace popart {

class Graph;
class Ir;

namespace liveness {

ExecutionContext sanitizeExecutionContext(ExecutionContext context);

class SubgraphCopyingStrategy;

// A vector of Op*s comprising a call stack.
using CallStack = std::vector<Op *>;

// OP type and function in the global schedule
enum class OpStatus {
  // Normal OP without subgraphs
  Normal = 0,
  // Subgraph entering
  Enter,
  CopyInput,
  // Subgraph loop carried
  CopyLoopCarried,
  // Subgraph exiting
  CopyOutput,
  CopyModified,
  Exit
};

class LivenessNode {
public:
  LivenessNode(const OpStatus status_,
               int index_,
               SubgraphIndex subgraphIndex,
               bool isDuplicate);

  LivenessNode(const CallStack &callStack_,
               const OpStatus status_,
               int index_,
               SubgraphIndex subgraphIndex,
               bool isDuplicate);

  // Operation associated with this node
  Op *getOp() const { return callStack.back(); }

  // Call stack of operators that call each other
  const CallStack &getCallStack() const { return callStack; }

  // Operation type and function in the global schedule
  OpStatus getStatus() const { return status; }

  // Input or output index of the Op associated with this node
  int getIndex() const { return index; }

  // For Enter, Exit, CopyInput, CopyOutput, CopyModified, CopyLoopCarried
  // nodes this signifies the subgraph called by the op.
  int getSubgraphIndex() const { return subgraphIndex; }

  // Determine if this node is a duplicate (duplicates are used for loops).
  bool getDuplicate() const { return isDuplicate; }

  // All tensor ids touched by this node
  const std::set<TensorId> &usedTensorIds() const { return usedIds; }

  // Pair of matching tensors inside/outside of the subgraph associated with the
  // index returned by getIndex on CopyOutput, CopyInput and CopyModified nodes
  const std::pair<TensorId, TensorId> &getTensorIds() const {
    return tensorIds;
  }

  // If the operation produces or fully modifies t
  bool isProducerOf(Tensor *t) const;

  // If the operation consumes t
  bool isConsumerOf(Tensor *t) const;

  // If the operation overwrites t
  bool overwritesTensor(Tensor *t) const;

  // If the operation modifies t
  bool modifiesTensor(Tensor *t) const;

  // Get the underlying, sanitized execution context
  ExecutionContext getExecutionContext() const {
    auto executionContext = callStack.back()->settings.executionContext;
    return sanitizeExecutionContext(executionContext);
  }

private:
  void setUsedTensorIds();
  void setTensorIds();

  CallStack callStack;
  OpStatus status;
  int index;
  SubgraphIndex subgraphIndex;
  bool isDuplicate;
  std::pair<TensorId, TensorId> tensorIds;
  std::set<TensorId> usedIds;
};

std::ostream &operator<<(std::ostream &os, const OpStatus &);
std::ostream &operator<<(std::ostream &os, const LivenessNode &);

// Build global schedule for liveness and call site analysis:
//
// A, B, C, D, E, ... : Non-subgraphing Op
// X: CallOp or LoopOp (subgraphing) type Op
// Y: IfOp (subgraphing) type Op
//
// + : OpStatus::Enter: subgraph enter
// - : OpStatus::Exit:  subgraph exit
// n : OpStatus::Normal: non-subgraphing op
//
// Global schedule example:
//                     1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 3 3 3 3
// 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3
// n n n +                   - n n n n n n +                   - n n n
// A B C X n n +     -     - X J K L M N O Z n n +     -     - Z P Q R
//         D E Y n n Y n n Y                 D E Y n n Y n n Y
//               F G   H I                         F G   H I
//
class LivenessAnalyzer {
public:
  // List of pending copies.
  using PendingCopies = std::vector<LivenessNode>;

  LivenessAnalyzer(const Ir *ir_,
                   const SubgraphCopyingStrategy *subgraphCopyingStrat);
  void apply();

  // For a given unique op call stack (e.g. X->Y->F),
  // return the global schedule position (e.g. 7)
  int64_t getGlobalSchedulePosition(CallStack callStack) const;

  // Size of the global schedule (e.g. 34)
  size_t getOpScheduleSize() const { return opSchedule.size(); }

  // Return the call stack (e.g. X->Y->F),
  // at a given schedule position (e.g. 7)
  const LivenessNode &getOpScheduleAt(int64_t scheduleIndex) const {
    return opSchedule.at(scheduleIndex);
  }

  // Get a graph's local schedule.
  const std::vector<Op *> &getGraphOpSchedule(GraphId id) const {
    return graphOpSchedule.at(id);
  }

  // Return all global schedule positions (e.g. 7, 24)
  // where an Op is called (e.g. F)
  const std::vector<int64_t> &getScheduleIndices(Op *op) const {
    return opScheduleMap.at(op);
  }

  // Return all global schedule positions where a tensor is used
  const std::vector<int64_t> &getScheduleIndices(Tensor *t) const {
    return tensorScheduleMap.at(t->id);
  }

  // Return all global schedule positions where a tensor is used
  const std::vector<int64_t> &getScheduleIndices(TensorId tid) const {
    return tensorScheduleMap.at(tid);
  }

  // Given the position of an OpStatus::Enter (e.g. 6)
  // return the matching OpStatus::Exit positions (e.g. 9, 12)
  const std::vector<int64_t> &getCallSiteLinksAt(int64_t scheduleIndex) const {
    return callSiteLinks.at(scheduleIndex);
  }

  // Inverse of getCallSiteLinksAt (give positions of exit given an enter).
  const std::vector<int64_t> &
  getCallSiteLinksInvAt(int64_t scheduleIndex) const {
    return callSiteLinksInv.at(scheduleIndex);
  }

  // Graph call sites in total order (e.g. for the graph called by Y: 6, 23)
  const std::vector<Op *> &getGraphCallSites(GraphId id) const {
    return graphCallSiteOps.at(id);
  }

  // Get the start position of a context, returns -1 if not found
  int64_t getContextStartIndex(ExecutionContext context) const;

  // Get the end position of a context, returns -1 if not found
  int64_t getContextEndIndex(ExecutionContext context) const;

private:
  // Global schedule including all subgraphs recursively
  void addToSchedule(const Graph *graphToAdd,
                     bool isDuplicate,
                     CallStack callStack,
                     PendingCopies &pendingCopies);

  // Add a single liveness node to the schedule. Returns offset into schedule.
  int64_t addNodeToSchedule(const LivenessNode &nodeToAdd,
                            PendingCopies &pendingCopies);

  // Process indices returned by SubgraphCopyStrategy.
  void
  processSubgraphCopyingStrategyIndices(PendingCopies &pendingCopies,
                                        std::vector<size_t> &chosenIndices);

  // Expand a subgraph.
  void expandSubgraph(const Graph *subgraph,
                      bool isDuplicate,
                      const CallStack &callStack,
                      SubgraphIndex subgraphIndex,
                      PendingCopies &pendingCopies);

  // Add a copies needed for subgraph to pendingCopies. The
  // SubgraphCopyingStrategy will be used to determine where they are added in
  // the schedule.
  void addCopiesToPending(const Graph *subgraph,
                          bool isDuplicate,
                          const CallStack &callStack,
                          SubgraphIndex subgraphIndex,
                          PendingCopies &pendingCopies);

  const Ir *ir;
  const SubgraphCopyingStrategy *subgraphCopyingStrat;

  // Mapping from graph id to the graph's final schedule
  std::map<GraphId, std::vector<Op *>> graphOpSchedule;

  // Global schedule (over all graphs) in final schedule order
  std::vector<LivenessNode> opSchedule;

  // Map of all schedule positions where an Op is called
  std::map<Op *, std::vector<int64_t>> opScheduleMap;

  // Map of all tensors and their usage location
  std::map<TensorId, std::vector<int64_t>> tensorScheduleMap;

  // Map of all OpStatus::Enter positions to the matching OpStatus::Exit
  // positions
  std::map<int64_t, std::vector<int64_t>> callSiteLinks;
  // Inverse of callSiteLinks
  std::map<int64_t, std::vector<int64_t>> callSiteLinksInv;

  // Map of all ops that call the graph referred to by GraphId
  std::map<GraphId, std::vector<Op *>> graphCallSiteOps;

  // Map of the context starts and ends
  std::map<ExecutionContext, int64_t> contextStarts;
  std::map<ExecutionContext, int64_t> contextEnds;
};

} // namespace liveness
} // namespace popart

#endif
