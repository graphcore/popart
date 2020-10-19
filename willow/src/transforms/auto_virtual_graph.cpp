// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <popart/error.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/logging.hpp>
#include <popart/names.hpp>
#include <popart/op.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorindex.hpp>
#include <popart/transforms/auto_virtual_graph.hpp>

using SubgraphId = int;

namespace popart {

std::pair<bool, OpId> Subgraph::best_split(float split_cost) {
  auto lb_node = split_nodes.lower_bound(split_cost);

  if (lb_node == split_nodes.end()) {
    return {false, -1};
  }

  auto best_node = lb_node;
  if (lb_node != split_nodes.begin()) {
    auto prev_lb_node = std::prev(lb_node);
    // Node closest to split_cost
    if ((split_cost - prev_lb_node->first) < (lb_node->first - split_cost))
      best_node = prev_lb_node;
  }

  split_nodes.erase(best_node);

  return {true, best_node->second};
}

std::size_t AutoVirtualGraph::id() {
  return typeid(AutoVirtualGraph).hash_code();
}

// The cost of an Op. Currently it's 1*input_weights + 1*outputs_to_grad + 0.
// This could be improved by having different parameters for each op type.
// This does not take into account Ops using the same Weights
float AutoVirtualGraph::costFn(Op *op,
                               bool training,
                               float w_weights     = 1.f,
                               float w_activations = 1.f) const {
  float total = 0;

  std::set<int> inputs_seen;
  std::set<int> outputs_seen;

  for (auto map : op->input->indicesMap()) {
    if (map.first->tensorType() == TensorType::Variable ||
        map.first->tensorType() == TensorType::Const) {
      total += w_weights * static_cast<float>(map.first->info.nbytes());
      // Record all input indices seen.
      for (int index : map.second) {
        inputs_seen.insert(index);
      }
    }
  }
  if (training) {
    // Check if backwards pass
    std::set<TensorId> backwardsTensors;
    for (auto &gradOp : op->getGradOps()) {
      for (auto &inOutMapper : gradOp->gradInputInfo()) {
        int indexFwd      = inOutMapper.iNonGrad;
        GradOpInType type = inOutMapper.type;
        // the input at index 'indexGrad' to gradOp is
        switch (type) {
        // An input to the fwd Op. Ignore weights seen previously.
        case GradOpInType::In: {
          bool exists = inputs_seen.insert(indexFwd).second;
          if (exists) {
            // This will need checking
            total +=
                w_activations *
                static_cast<float>(op->input->tensor(indexFwd)->info.nbytes());
          }
          break;
        }

        //  An output from the fwd Op.
        case GradOpInType::Out: {
          bool exists = outputs_seen.insert(indexFwd).second;
          if (exists) {
            total +=
                w_activations *
                static_cast<float>(op->output->tensor(indexFwd)->info.nbytes());
          }
          break;
        }

        // This is the data that passes through the backwards pass.
        // Unless the VarUpdate is done as a single compute_set
        // This input can be ignored as not 'always live'
        case GradOpInType::GradOut: {
          break;
        }
        }
      }
    }
  }
  return total;
}

// Splits the graph for model parallelism. To do this it needs to do 3 things:
// 1) Find potential split nodes.
// 2) Calculate a cost model for each split node
// 3) Select a set of nodes which balance the costs across each processor.
//
//   A   B      Subgraphs:
//   |   |        {A,C}, {B}, {D,E}
//   C   |
//   \   /  ->  If splitting for 2 IPUs:
//     D          IPU0: {A,C}. IPU1: {B,D,E}
//     |        If splitting for 4 IPUs:
//     E          IPU0: {A}. IPU1: {C}. IPU2: {B}. IPU3: {D,E}
//
// 1.1) Separate graph into subgraphs
//   - Create a subgraph for each consumer of each dataStreamTensor (Inputs to
//     graph)
//   - Add all consuming Ops to its parent subgraph
//   - If an Op consumes from more than one subgraph, create a new subgraph for
//     it.
// 1.2) Find split nodes in subgraphs
//   - Find nodes in the subgraph where all data collapses into a single Op.
// 2) Calculate a cost for Op.
//   - Keep a cumulative cost of the whole graph.
//   - Keep a cumulative cost of each subgraph.
//   - When saving a potential split node, save the cumulative cost of that
//     subgraph up to the node.
// 3) Subgraphs have been created in topological order, so..
//   - Place subgraphs on a virtualGraph starting with 0.
//   - If the total cost of a subgraph is more than the desired
//     proportion of the total cost on a single IPU, find a split in the
//     subgraph and place subsequent ops on the next virtualGraph.
bool AutoVirtualGraph::apply(Graph &graph) const {
  auto &ir = graph.getIr();

  auto &opts = ir.getSessionOptions();
  auto replicationDivisor =
      opts.enableReplicatedGraphs ? opts.replicatedGraphCount : 1;
  const auto num_ipus = ir.getDeviceInfo()->getNumIpus() / replicationDivisor;
  const auto training = ir.canTrain();

  if (graph.getOps().size() == 0 || num_ipus < 2) {
    return true;
  }

  float w_weights = 1.0f;
  if (opts.enableGradientAccumulation) {
    // Weights are doubled as there is an accumulator to match each.
    w_weights = 2.0f;
  }

  logging::transform::info("[AutoVirtualGraph] Auto virtual graph with {} IPUs",
                           num_ipus);

  float cumulative_cost = 0.f;
  std::vector<Subgraph> subgraphs;
  std::map<OpId, SubgraphId> node_subgraph_map;

  SubgraphId next_subgraph_id = 0;

  auto startNewSubgraph =
      [&node_subgraph_map, &subgraphs, &next_subgraph_id](OpId conId) {
        auto iter = node_subgraph_map.insert({conId, next_subgraph_id});
        subgraphs.push_back({conId});
        next_subgraph_id++;
        return iter;
      };

  for (auto *t : ir.dataStreamTensors()) {
    for (Op *consumer_op : t->consumers.getOps()) {
      startNewSubgraph(consumer_op->id);
      logging::transform::trace(
          "Starting at {} {}.", consumer_op->debugName(), consumer_op->id);
    }
  }
  // In topological order...
  auto schedule = graph.getOpSchedule({}, RequireOptimalSchedule::Yes);

  for (Op *op : schedule) {

    if (op->toLoss != PathToLoss::Undefined) {
      throw internal_error(
          "Op {} has been annotated with PathToLoss "
          "information, AutoVirtualGraph::apply should be applied "
          "before the final loss is grown though",
          op->str());
    }

    // Find potential split nodes
    float op_cost = costFn(op, training, w_weights);

    // Keep a cumulative_cost of the whole graph.
    cumulative_cost += op_cost;

    // If the Op has a path to it from a Stream Tensor, it will have been
    // assigned a sub-graph
    auto subgraph_id_found = node_subgraph_map.find(op->id);

    // If the Op does not have a path to it from a Stream Tensor, it will not
    // yet have been assigned a sub-graph
    if (subgraph_id_found == node_subgraph_map.end()) {
      subgraph_id_found = startNewSubgraph(op->id).first;
    }

    auto subgraph_id = subgraph_id_found->second;
    auto &subgraph   = subgraphs.at(subgraph_id);

    // Keep a cumulative cost of each subgraph.
    subgraph.cost += op_cost;

    // Op is a split in the subgraph, following rule 1.1 (see above)
    if (subgraph.candidates.erase(op->id) && subgraph.candidates.empty()) {
      subgraph.split_nodes.insert({subgraph.cost, op->id});
      logging::transform::trace(
          "Adding split on subgraph {}, node {} {}, cost {}",
          subgraph_id,
          op->debugName(),
          op->id,
          subgraph.cost);
    }

    // Add Op consumers to the subgraph candidates.
    for (Tensor *t : op->output->tensors()) {
      for (Op *consumer_op : t->consumers.getOps()) {
        auto insert_result =
            node_subgraph_map.insert({consumer_op->id, subgraph_id});
        // If this op has been seen in another subgraph. Create a new
        // subgraph for it.
        if (!insert_result.second &&
            insert_result.first->second != subgraph_id) {
          auto &conflict_subgraph = subgraphs.at(insert_result.first->second);
          if (conflict_subgraph.candidates.size() == 1 &&
              conflict_subgraph.cost == 0.0f) {
            // This subgraph is a single op and has no cost, so don't create
            // a new subgraph Just move it into the current op's subgraph.
            logging::transform::trace(
                "Moving node {} {} from subgraph {} to {}.",
                consumer_op->debugName(),
                consumer_op->id,
                insert_result.first->second,
                subgraph_id);
            conflict_subgraph.candidates.erase(consumer_op->id);
            subgraph.candidates.insert(consumer_op->id);
            insert_result.first->second = subgraph_id;
          } else {
            logging::transform::trace("Creating new subgraph {} at {} {}.",
                                      next_subgraph_id,
                                      consumer_op->debugName(),
                                      consumer_op->id);
            subgraphs.push_back({consumer_op->id});
            insert_result.first->second = next_subgraph_id;
            next_subgraph_id++;
          }
        } else {
          subgraph.candidates.insert(consumer_op->id);
        }
      }
    }
  }

  SubgraphId id = 0;
  for (auto subgraph : subgraphs) {
    logging::transform::trace("Subgraph {} cost {} splits {}",
                              id,
                              subgraph.cost,
                              subgraph.split_nodes.size());
    id++;
  }
  logging::transform::trace("Total graph cost {}", cumulative_cost);

  // Find best splits for the number of ipus
  float total_subgraph_costs = 0;
  int64_t virtual_graph_id   = 0;
  SubgraphId subgraph_id     = 0;
  for (size_t i = 1; i <= num_ipus; i++) {
    logging::transform::trace(
        "[AutoVirtualGraph] Considering what to assign to IPU: {}", i - 1);
    float split_ratio = float(i) / float(num_ipus);
    float split_cost  = split_ratio * cumulative_cost;
    while (subgraph_id < subgraphs.size()) {
      auto &subgraph = subgraphs.at(subgraph_id);

      // How do we assign the current subgraph to current IPU?
      // In one of four ways:
      // 1) Fill budget for this IPU by assigning a split of this subgraph
      // 2) Fill budget for this IPU by assigning rest of this subgraph
      // 3) Fill budget for this IPU by assigning entirety of this subgraph
      // 4) Assign entirety of this subgraph to this IPU, with room to spare
      // The below algorithm covers there four cases together.

      bool above_ipu_cost_budget =
          (subgraph.cost + total_subgraph_costs) > split_cost;

      // Design decision: Always put a zero-cost subgraph on the current IPU.
      // It is possible that assigning the previous subgraph split has already
      // taken us over the cost budget for this IPU. If this subgraph were an
      // empty subgraph (therefore with no valid splits) we would end up with
      // an empty IPU. Always putting a zero-cost subgraph on the current IPU
      // avoids this issue. TODO T13956 : Avoid this problem more elegantly
      above_ipu_cost_budget = above_ipu_cost_budget && (subgraph.cost > 0);

      bool split_at_end =
          (subgraph.cost + total_subgraph_costs) - split_cost == 0.0f;
      if (above_ipu_cost_budget && i != num_ipus) {
        // Subgraph is bigger than the desired split
        // Find a split in the subgraph that best matches
        // the amount needed to reach split_cost
        // But not if it's the last IPU
        auto split = subgraph.best_split(split_cost - total_subgraph_costs);

        if (split.first) {
          // A split has been found
          subgraph.split_nodes.erase(split.second);
          subgraph.final_splits.insert(split.second);
          logging::transform::trace(
              "[AutoVirtualGraph]   Assigning a split of subgraph: {}",
              subgraph_id);
          logging::transform::info("[AutoVirtualGraph]   Split node: {}",
                                   graph.getOp(split.second)->debugName());

          // Does the virtual_graph_id need setting?
          if (!subgraph.has_been_split) {
            subgraph.virtual_graph_id = virtual_graph_id;
          }
          subgraph.has_been_split = true;

          virtual_graph_id++;
          break;
        } else {
          // Could not find split node so place it at the end of the subgraph.
          logging::transform::trace(
              "[AutoVirtualGraph]   Could not find split node of cost {}",
              split_cost - total_subgraph_costs);
          split_at_end = true;
        }
      }
      // The whole the subgraph consumed.
      total_subgraph_costs += subgraph.cost;
      // Skip over this subgraph next split_cost
      subgraph_id++;

      // Does the virtual_graph_id need setting?
      if (!subgraph.has_been_split) {
        subgraph.virtual_graph_id = virtual_graph_id;
      }

      // Split at the end of the subgraph
      // But not if it's the end of the graph
      if (split_at_end && i != num_ipus && subgraph_id != subgraphs.size()) {
        logging::transform::info("[AutoVirtualGraph]   Assigning remainder of "
                                 "subgraph: {}. Split at end",
                                 subgraph_id - 1);
        virtual_graph_id++;
        break;
      }
    }
  }

  if (virtual_graph_id != num_ipus - 1) {
    throw error("[AutoVirtualGraph] Couldn't find enough splits for {} IPUs. "
                "Only found {}",
                num_ipus,
                virtual_graph_id);
  }

  // Add sharding information to graph.
  // Note, the transform did not mutate the graph, so we can reuse the schedule
  // we computed before.
  for (Op *op : schedule) {
    // Find potential split nodes
    auto &subgraph = subgraphs.at(node_subgraph_map.find(op->id)->second);
    op->setVirtualGraphId(subgraph.virtual_graph_id);
    if (subgraph.final_splits.erase(op->id)) {
      // Does the op go on the previous or next graph? For now previous.
      subgraph.virtual_graph_id++;
    }
  }

  return true;
}

namespace {
bool init = Transform::registerTransform(new AutoVirtualGraph);
}

} // namespace popart
