// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_LIVENESS_HPP
#define GUARD_NEURALNET_LIVENESS_HPP

#include <popart/graphid.hpp>
#include <popart/op.hpp>

namespace popart {

class Graph;

namespace liveness {

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
  LivenessNode(const OpStatus status_, int index_);

  LivenessNode(std::vector<Op *> callStack_,
               const OpStatus status_,
               int index_);

  // Operation associated with this node
  Op *getOp() const { return callStack.back(); }

  // Call stack of operators that call each other
  const std::vector<Op *> &getCallStack() const { return callStack; }

  // Operation type and function in the global schedule
  OpStatus getStatus() const { return status; }

  // Input or output index of the Op associated with this node
  int getIndex() const { return index; }

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

private:
  void setUsedTensorIds();
  void setTensorIds();

  std::vector<Op *> callStack;
  OpStatus status;
  int index;
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
  LivenessAnalyzer(const Ir *ir_);
  void apply();

  // For a given unique op call stack (e.g. X->Y->F),
  // return the global schedule position (e.g. 7)
  int64_t getGlobalSchedulePosition(std::vector<Op *> callStack) const;

  // Size of the global schedule (e.g. 34)
  size_t getOpScheduleSize() const { return opSchedule.size(); }

  // Return the call stack (e.g. X->Y->F),
  // at a given schedule position (e.g. 7)
  const LivenessNode &getOpScheduleAt(int64_t scheduleIndex) const {
    return opSchedule.at(scheduleIndex);
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

  // Graph call sites in total order (e.g. for the graph called by Y: 6, 23)
  const std::vector<Op *> &getGraphCallSites(GraphId id) const {
    return graphCallSiteOps.at(id);
  }

private:
  // Global schedule including all subgraphs recursively
  void addToSchedule(const Graph *, std::vector<Op *>);

  const Ir *ir;

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

  // Map of all ops that call the graph referred to by GraphId
  std::map<GraphId, std::vector<Op *>> graphCallSiteOps;
};

} // namespace liveness
} // namespace popart

#endif
