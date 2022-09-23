// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OP_LOOP_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_LOOP_HPP_

#include <functional>
#include <memory>
#include <set>
#include <vector>
#include <popart/op.hpp>
#include <popart/op/subgraph.hpp>
#include <popart/tensorlocation.hpp>

#include "popart/names.hpp"

namespace popart {

struct OperatorIdentifier;
class AliasModel;
class Graph;
class OpSerialiserBase;

// Loop operation construct
//
// Loop op and body inputs:
// 0    trip count                       can be omitted as Op input
// 1    termination condition <--.       can be omitted as Op input
// 2    explicit input <---------|--.
// ..   ..             ..        |  |
// M    explicit input <---------|--|--.
// M+1  implicit input           |  |  |
// ..   ..                       |  |  |
// N    implicit input           |  |  |
//                               |  |  | Loop carried dependencies
// Loop body outputs:            |  |  |
// 0     termination condition --'  |  |
// 1     output --------------------+  |
// ..    ..                         |  |
// M-1   output --------------------|--+
// M     implicit scan output ------|--|-----.
// ..    ..                         |  |     |
// K     implicit scan output ------|--|--.  |
//                                  |  |  |  |
// Loop op outputs:                 |  |  |  |
// 0     output <-------------------'  |  |  |
// ..    ..                            |  |  |
// M-2   output <----------------------'  |  |
// M-1   implicit scan output <-----------'  |  } Only supported pre
// ..    ..                                  |  } LoopScanOutPattern
// K-1   implicit scan output <--------------'  }
//
//
// Explicit inputs are loop-carried inputs that are copied into the loop body,
// copied from loop body output to loop body input after each iteration,
// and copied out of the loop into the parent graph as loop outputs on
// loop termination.
//
// Implicit inputs are only copied into the loop body once, and are implicit
// because in the ONNX specification, the loop body can consume tensors from
// the parent graph without it being a subgraph input to the loop body.
// In the PopART IR, we add implicit inputs to the LoopOp and loop body
// subgraph, due to subgraphs not being allowed to consume parent graph tensors
// directly.
// This also allows loop body subgraphs to be reused at multiple call sites.
//
//
// Note:
// The trip count and termination condition tensors are optional as Op inputs
// The condition tensor input/output in the subgraph needs to exist in any case

class LoopOp : public SubgraphOp {
public:
  LoopOp(const OperatorIdentifier &, const Op::Settings &, Graph &callee_);

  LoopOp(const OperatorIdentifier &,
         const Op::Settings &,
         Graph &callee_,
         int numImplicitScanOutputs_);

  void setup() final;
  void appendOutlineAttributes(OpSerialiserBase &) const override;
  void connectInTensor(InIndex inIndex, TensorId tensorId) final;
  float getSubgraphValue() const final { return getLowSubgraphValue(); }
  std::unique_ptr<Op> clone() const override;
  std::vector<const Graph *> getCalledGraphs() const final;
  std::vector<TensorId> implicitInputTensors() const;

  Graph &getCalledGraph() const override;
  void setCalledGraph(Graph &) override;

  int getTripCountValue() const { return tripCountValue; }
  void setTripCountValue(int value) { tripCountValue = value; }
  int getNumExplicitInputs() const;
  int getNumImplicitInputs() const;
  int getNumImplicitScanOutputs() { return numImplicitScanOutputs; }
  void setNumImplicitScanOutputs(int numOutputs) {
    numImplicitScanOutputs = numOutputs;
  }

  InIndex subgraphInToOpInIndex(InIndex index) const override;
  InIndex opInToSubgraphInIndex(InIndex index) const override;

  OutIndex subgraphOutToOpOutIndex(OutIndex index) const override;
  OutIndex opOutToSubgraphOutIndex(OutIndex index) const override;

  VGraphIdAndTileSet
  getIntrospectionInVirtualGraphId(InIndex,
                                   std::set<OpId> &visited) const override;
  VGraphIdAndTileSet
  getIntrospectionOutVirtualGraphId(OutIndex,
                                    std::set<OpId> &visited) const override;

  /**
   * Indexing on the LoopOp
   * \return The LoopOp input index for the maximum number of loop iterations
   */
  static InIndex getMaximumTripCountInIndex() { return 0; }

  /**
   * Indexing on the LoopOp
   * \return The LoopOp input index specifying the termination condition status
   */
  static InIndex getTerminationConditionInIndex() { return 1; }

  /**
   * Indexing on the LoopOp
   * \return The first regular, user-defined LoopOp input index
   */
  static InIndex getFirstInputInIndex() { return 2; }

  /**
   * Indexing on the LoopOp
   * \return The first regular, user-defined LoopOp output index
   */
  static OutIndex getFirstOutputOutIndex() { return 0; }

  /**
   * Indexing on the body graph
   * \return The loop body graph input index specifying the current loop
   *         iteration
   */
  static InIndex getLoopGraphIterationInIndex() { return 0; }

  /**
   * Indexing on the body graph
   * \return The loop body graph input index specifying the current termination
   *         condition status
   */
  static InIndex getLoopGraphTerminationConditionInIndex() { return 1; }

  /**
   * Indexing on the body graph
   * \return The first regular, user-defined loop body graph input index
   */
  static InIndex getLoopGraphFirstInputInIndex() { return 2; }

  /**
   * Indexing on the body graph
   * \return The loop body graph output index for the termination condition
   *         status after the loop body graph has run
   */
  static OutIndex getLoopGraphTerminationConditionOutIndex() { return 0; }

  /**
   * Indexing on the body graph
   * \return The first regular, user-defined loop body graph output index
   */
  static OutIndex getLoopGraphFirstOutputOutIndex() { return 1; }

  /**
   * Add a variadic input to the loop operator.
   *
   * \param index The position at which a Tensor is consumed by the Op.
   * \param tensorId The id of the tensor to add as an input.
   * \param subgraphTensorId Tensor which is going to be created in the
   *                         subgraph.
   * \param overwrite If true the original tensor at `index` will be
   *                  replaced.
   **/
  void addLoopInput(InIndex index,
                    TensorId tensorId,
                    TensorId subgraphTensorId,
                    bool overwrite);

  void addLoopOutput(OutIndex index,
                     TensorId tensorId,
                     TensorId subgraphTensorId,
                     bool overwrite);

  void removeLoopInput(InIndex index);
  void removeLoopOutput(OutIndex index);

  void growAliasModel(AliasModel &m) const override { growAliasModelMulti(m); }

  std::set<OutIndex> opInToOpOutIndex(InIndex in) const override;
  std::set<InIndex> opOutToOpInIndex(OutIndex out) const override;

private:
  std::reference_wrapper<Graph> callee;
  int tripCountValue;
  int numImplicitScanOutputs;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_OP_LOOP_HPP_
