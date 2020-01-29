#include <popart/error.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/names.hpp>
#include <popart/op.hpp>
#include <popart/op/cache.hpp>
#include <popart/op/call.hpp>
#include <popart/op/init.hpp>
#include <popart/op/ipucopy.hpp>
#include <popart/tensor.hpp>
#include <popart/tensors.hpp>
#include <popart/topocons.hpp>
#include <popart/transforms/aliaszerocopy.hpp>

namespace popart {

std::size_t AliasZeroCopy::id() { return typeid(AliasZeroCopy).hash_code(); }

// TODO clean up this function (what does it do exactly?)
bool AliasZeroCopy::isSameTensor(Graph &graph,
                                 std::vector<Tensor *> tensors) const {
  if (tensors.size() <= 1)
    return true;

  auto t0     = tensors.front();
  auto t0full = view::Region::getFull(t0->info.shape());
  for (Tensor *t1 : tensors) {
    if (t1->id != t0->id) {
      // Regions in t0 which t1 aliases
      auto regions = graph.getTensors().getAliasRegions(t1, t0);
      if (std::any_of(
              regions.begin(), regions.end(), [&t0full](const view::Region &r) {
                return r == t0full;
              })) {
        // t1 is fully aliased to t0
      } else {
        return false;
      }
    }
  }
  return true;
}

// Glossary:
// P: Producer
// C: Consumer
// U: Updating/aliasing operation
// t: Tensor
// (R): Recomputation

// If P1/P2/P3 are InitOps and t1/t3/t5 can be aliased
// (are not used in overlapping intervals),
// and t1/t2/t7, t3/t4, t5/t6 are already aliased:
//
//                    /----------------- C1(R) --- t8(R)
// P1 -- t1 -- U1 -- t2 -- C1 -- t8                |
//                    \--------------- U4 -- t7 -- C4
// P2 -- t3 -- U2 -- t4 -- C2
// P3 -- t5 -- U3 -- t6 -- C3

// After transforming, would look like this:
//
//                    /----------------------------------------- C1(R) --- t8(R)
// P1 -- t1 -- U1 -- t2 -- C1 -- t8                                        |
//       |            \------------------------------------- U4 ---- t7 -- C4
//       \-------------------- U2 -- t4 -- C2
//        \----------------------------------- U3 -- t6 -- C3
// Now, if U1, U2 and U3 are the same operator, they are outlined,
// and a zero-copy for the outlined subgraph is possible, since t1 and t2/t4/t6
// all refer to the same tensor (aliased under multiple names).

// If t2 is required again in the recomputation of C1, then this is not a
// problem, as long as U4 restores t2, aliased as t7, before C1(R) is executed.

bool AliasZeroCopy::apply(Graph &graph) const {
  logging::debug("[AliasZeroCopy] Started.");

  auto schedule = graph.getOpSchedule({});

  std::map<Op *, int64_t> scheduleLookup;
  std::map<GraphId, std::vector<CallOp *>> candidates;

  for (int64_t i = 0; i < schedule.size(); ++i) {
    Op *op             = schedule[i];
    scheduleLookup[op] = i;
    if (CallOp *callOp = dynamic_cast<CallOp *>(op)) {
      GraphId id = callOp->getCalledGraph().id;
      candidates[id].push_back(callOp);
    }
  }

  for (auto &candidate : candidates) {
    auto num_inputs  = candidate.second[0]->input->n();
    auto num_outputs = candidate.second[0]->output->n();

    for (int j = 0; j < num_outputs; ++j) {
      // Output aliasing
      std::map<int64_t, std::vector<Op *>> map;
      bool aliasPossible = true;
      for (CallOp *op : candidate.second) {
        // TODO: Liveness analysis instead of pingpong phase
        map[op->getPingPongPhase()].push_back(op);
        if (map[op->getPingPongPhase()].size() > 1) {
          logging::trace("No aliasing for subgraph {}, output {} possible, "
                         "because of producers in the same phase: {}.",
                         candidate.second[0]->getCalledGraph().id,
                         j,
                         op->getPingPongPhase());
          aliasPossible = false;
          break;
        }
      }
      if (aliasPossible) {
        logging::debug("Doing aliasing for subgraph {}, output {}",
                       candidate.second[0]->getCalledGraph().id,
                       j);
        TensorId aliasingTensor;
        /*
        for (auto &elem : map) {
          if (aliasingTensor.size() == 0) {
            // Op *op = elem.second[0];
            // aliasingTensor = op->input->id(j);
          }
        }
        */
      }
    }

    for (int j = 0; j < num_inputs; ++j) {
      std::vector<Tensor *> tensors;
      tensors.reserve(candidate.second.size());

      // Input zero copy
      for (CallOp *op : candidate.second) {
        tensors.push_back(op->input->tensor(j));
      }
      std::map<int64_t, std::pair<Op *, Tensor *>> producerPriorityMap;
      bool zeroCopyPossible = true;
      if (!isSameTensor(graph, tensors)) {
        for (size_t k = 0; k < tensors.size(); ++k) {
          auto tensor = tensors[k];
          Op *op      = tensor->getProducerUnsafe();
          if (op &&
              (dynamic_cast<CacheLoadOp *>(op) || dynamic_cast<CallOp *>(op) ||
               dynamic_cast<InitOp *>(op))) {
            producerPriorityMap[scheduleLookup[op]] = {op, tensor};
            std::pair<int64_t, int64_t> interval{
                scheduleLookup[op], scheduleLookup[candidate.second[k]]};
          } else {
            logging::trace("[AliasZeroCopy] No zero copy for subgraph {}, "
                           "input {} possible, because producer is of type {}.",
                           candidate.second[0]->getCalledGraph().id,
                           j,
                           tensor->hasProducer()
                               ? tensor->getProducer()->debugName()
                               : "");
            zeroCopyPossible = false;
            break;
          }
        }
      } // !isSameTensor

      if (zeroCopyPossible) {
        logging::debug(
            "[AliasZeroCopy] Doing zero copy for subgraph {}, input {}, "
            "{} producers",
            candidate.second[0]->getCalledGraph().id,
            j,
            producerPriorityMap.size());

        TensorId current_id;
        for (auto &producerEntry : producerPriorityMap) {
          Op *op         = producerEntry.second.first;
          Tensor *tensor = producerEntry.second.second;

          logging::trace(
              "[AliasZeroCopy] Working on producer {}, out indices {}",
              op->debugName(),
              op->output->indices(tensor));

          if (current_id.size() == 0) {
            // Starting the aliasing chain
            current_id = tensor->id;
          } else {
            // Continuing the aliasing chain
            if (InitOp *cacheAlloc = dynamic_cast<InitOp *>(op)) {
              // Special case (currently sufficient for weight tensors),
              // where the producer is a InitOp, which can always be
              // removed safely.
              auto consumers = tensor->consumers.getOps();
              // Currently, first consumer is guaranteed to be the updating op
              // FIXME: Might not always be the case.
              std::map<int64_t, Op *> orderedConsumers;
              for (Op *consumer : consumers) {
                auto it = std::find(schedule.begin(), schedule.end(), consumer);
                orderedConsumers[std::distance(schedule.begin(), it)] =
                    consumer;
              }

              bool first = true;
              for (auto &consumer : orderedConsumers) {
                Op *consumerOp = consumer.second;
                logging::trace("[AliasZeroCopy] Working on consumer {}",
                               consumerOp->debugName());
                if (first) {
                  // Updating op
                  auto inIndices = consumerOp->input->indices(tensor);

                  consumerOp->disconnectInTensor(tensor);
                  for (auto index : inIndices) {
                    consumerOp->connectInTensor(index, current_id);
                  }
                  consumerOp->setup();

                  op->disconnectAllInputs();
                  op->disconnectAllOutputs();
                  graph.eraseOp(op->id);
                  first = false;
                } else {
                  // Remaining consumers
                  throw error("[AliasZeroCopy] Only expected one consumer "
                              "of tensor {}",
                              tensor->id);
                }
              }

            } else {
              // Normal case
              // TODO
            }
          }
        }

        // Replace subgraph input with zero-copy (reference)
        auto &calledGraph     = candidate.second[0]->getCalledGraph();
        TensorId subgraphInId = calledGraph.getInputId(j);
        calledGraph.markAsZeroCopy(subgraphInId);
      }
    }
  }

  logging::debug("[AliasZeroCopy] Done.");
  return true;
}

namespace {
// AliasZeroCopy
bool init = Transform::registerTransform(new AliasZeroCopy());
} // namespace

} // namespace popart
