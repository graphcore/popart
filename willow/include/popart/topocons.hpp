// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_TOPOCONS_HPP
#define GUARD_NEURALNET_TOPOCONS_HPP

#include <popart/names.hpp>
#include <popart/op.hpp>
#include <popart/pointercomparators.hpp>

namespace popart {
// Topological constraints
//
// A topological constraint is a single edge between
// 2 Ops, stating the relative order they must appear
// in when scheduled (topologically sorted).
// we use a -> b to denote tnat a must be scheduled before b.
// These constraints are needed to support ops
// where input and output tensors share memory, such as
// in-place ops, view-changing ops, and weight-update ops.

class TopoOp {
public:
  TopoOp(Op *op_, bool tied_) : op(op_), tied(tied_) {}
  TopoOp(Op *op_) : op(op_), tied(false) {}
  TopoOp() = delete;
  Op *op;
  bool tied;
  bool operator<(const TopoOp &rhs) const { return op->id < rhs.op->id; }
};

class TopoCons {
public:
  // remove all topological constraints with op in it
  void remove(Op *op);
  void remove(Op *before, Op *after);

  // insert the constraint "before -> after"
  // if already present, do nothing
  void insert(Op *before, Op *after, bool tied = false);
  void insert(const OpsBeforeKey &, bool tied = false);

  // replace all topological constraints involving "beforeTransfer"
  // with "afterTransfer", on both ends of topological constraints
  void transfer(Op *beforeTransfer, Op *afterTransfer, bool removeOld = true);

  // Returns true if there is a constraints on this operation
  bool hasConstraint(Op *op);

  // replace each topological constraint involving "beforeTransfer"
  // with afterTransfer.size() constraints, one for each element of
  // afterTransfer
  void transferToMultiple(Op *beforeTransfer,
                          const std::vector<Op *> &afterTransfer,
                          bool removeOld = true);

  // replace each topological constraint from opRemap to either the replacement
  // Op (the subgraphing Op, external topocons),
  // or the subgraph Ops (internal topocons of the subgraph)
  // Before (arrows indicate topocons between Ops):
  //    A -> B -> C -> D
  //         ~    ~ (move into subgraph)
  //
  // After (E replaces B, C):
  //    A -> E -> D
  //    E: B -> C
  static void
  transferToSubgraph(Op *replacementOp,
                     std::map<Op *, std::vector<Op *>, POpCmp> opRemaps,
                     bool removeOld = true);

  bool contains(Op *before, Op *after) const;
  std::vector<Op *> getAfters(Op *before) const;
  std::vector<Op *> getBefores(Op *after) const;

  std::vector<Op *> getTiedAfters(Op *before) const;
  std::vector<Op *> getTiedBefores(Op *after) const;

  // required topological constraints such that "last"
  // is guaranteed to run after all other consumers
  // of Tensor "consumed"
  OpsBeforeKey finalConsumerCons(const Tensor *consumed, Op *last) const;

  friend std::ostream &operator<<(std::ostream &os, const TopoCons &tc);

  const std::map<Op *, std::set<TopoOp>, POpCmp> &getValsAfter() const {
    return valsAfter;
  }
  const std::map<Op *, std::set<TopoOp>, POpCmp> &getValsBefore() const {
    return valsBefore;
  }

private:
  // for all val : set, "key -> val"
  std::map<Op *, std::set<TopoOp>, POpCmp> valsAfter;

  // the mirror of valsAfterKey, so for all val : set, "val -> key"
  std::map<Op *, std::set<TopoOp>, POpCmp> valsBefore;
};

std::ostream &operator<<(std::ostream &os, const TopoCons &tc);

} // namespace popart

#endif
