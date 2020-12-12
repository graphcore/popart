// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <memory>
#include <queue>
#include <popart/error.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/names.hpp>
#include <popart/op.hpp>
#include <popart/op/dynamic/dynamicslice.hpp>
#include <popart/op/dynamic/dynamicupdate.hpp>
#include <popart/op/identity.hpp>
#include <popart/op/init.hpp>
#include <popart/op/loop.hpp>
#include <popart/op/reshape.hpp>
#include <popart/tensor.hpp>
#include <popart/tensors.hpp>
#include <popart/topocons.hpp>
#include <popart/transforms/mergeloops.hpp>
#include <popart/transforms/prune.hpp>

namespace popart {

bool MergeLoops::canMerge(const LoopOp *const op0,
                          const LoopOp *const op1) const {

  if (op0->scheduledPreLoss != op1->scheduledPreLoss) {
    return false;
  }

  if (op0->getTripCountValue() != op1->getTripCountValue()) {
    return false;
  }

  if (op0->getOptionalExecutionPhase() != op1->getOptionalExecutionPhase()) {
    return false;
  }

  if (op0->getOptionalPipelineStage() != op1->getOptionalPipelineStage()) {
    return false;
  }

  if (op0->hasInput(LoopOp::getTerminationConditionInIndex()) !=
      op1->hasInput(LoopOp::getTerminationConditionInIndex())) {
    return false;
  }

  if (op0->hasInput(LoopOp::getMaximumTripCountInIndex()) !=
      op1->hasInput(LoopOp::getMaximumTripCountInIndex())) {
    return false;
  }

  if (op0->hasInput(LoopOp::getTerminationConditionInIndex()) &&
      op0->input->id(LoopOp::getTerminationConditionInIndex()) !=
          op1->input->id(LoopOp::getTerminationConditionInIndex())) {
    return false;
  }

  if (op0->hasInput(LoopOp::getMaximumTripCountInIndex()) &&
      op0->input->id(LoopOp::getMaximumTripCountInIndex()) !=
          op1->input->id(LoopOp::getMaximumTripCountInIndex())) {
    return false;
  }

  return true;
}

bool MergeLoops::canMerge(const std::vector<LoopOp *> loopOps,
                          const LoopOp *const loopOp0) const {
  const Ir &ir = loopOp0->getIr();

  // Check basic compatibility of the LoopOps
  bool merge = true;
  for (LoopOp *loopOp1 : loopOps) {
    merge &= canMerge(loopOp0, loopOp1);
    if (!merge) {
      logging::trace("[MergeLoops] cannot merge {} -> {} (incompatible loops)",
                     loopOp0->debugName(),
                     loopOp1->debugName());
    }
  }
  if (!merge) {
    return false;
  }

  // Check that:
  // 1.) There is no indirect path from the existing set of LoopOps to the input
  // 2.) If the input is a direct output of the existing set of LoopOps,
  //     the sharding/concatenation path inside the mergeable LoopOps cancel out
  for (auto &indexAndTensor : loopOp0->input->tensorMap()) {
    Tensor *t = indexAndTensor.second;
    if (t->hasProducer() && t->getProducer()->isConvertibleTo<LoopOp>() &&
        std::find(loopOps.begin(),
                  loopOps.end(),
                  dynamic_cast<LoopOp *>(t->getProducer())) != loopOps.end()) {
      // 1.) Direct path
      LoopOp *loopOp1 = dynamic_cast<LoopOp *>(t->getProducer());

      // 2.) Resolve sharding/concatenation paths
      OutIndex outIdx = loopOp1->output->indices(t).front();
      InIndex inIdx   = indexAndTensor.first;

      TensorId sgOutId = loopOp1->getCalledGraph().getOutputId(
          loopOp1->opOutToSubgraphOutIndex(outIdx));
      TensorId sgInId = loopOp0->getCalledGraph().getInputId(
          loopOp0->opInToSubgraphInIndex(inIdx));

      Tensor *sgOut = ir.getTensor(sgOutId);
      Tensor *sgIn  = ir.getTensor(sgInId);

      // Resolved: If condition 2.) is satisfied and the paths cancel out
      bool resolved = false;
      std::queue<std::pair<Tensor *, Tensor *>> queue;
      queue.push({sgOut, sgIn});
      while (!queue.empty()) {
        auto tensors = queue.front();
        queue.pop();
        Op *producer = nullptr;
        Op *consumer = nullptr;
        if (tensors.first->hasProducer()) {
          producer = tensors.first->getProducer();
        }
        if (tensors.second->consumers.getOps().size() == 1) {
          consumer = tensors.second->consumers.getOps().front();
        }
        if (producer && consumer) {
          logging::transform::trace("[MergeLoops] Testing path {} {}",
                                    producer->debugName(),
                                    consumer->debugName());

          // Try to see if producer & consumer cancel each other out
          if (producer->isConvertibleTo<ReshapeOp>() &&
              consumer->isConvertibleTo<ReshapeOp>()) {
            Tensor *in  = producer->inTensor(ReshapeOp::getInIndex());
            Tensor *out = consumer->outTensor(ReshapeOp::getOutIndex());
            if (in->info == out->info) {
              queue.push({in, out});
            }
          } else if (producer->isConvertibleTo<DynamicUpdateOp>() &&
                     consumer->isConvertibleTo<DynamicSliceOp>()) {

            DynamicUpdateOp *dproducer =
                dynamic_cast<DynamicUpdateOp *>(producer);
            DynamicSliceOp *dconsumer =
                dynamic_cast<DynamicSliceOp *>(consumer);

            Tensor *inIdx =
                producer->inTensor(DynamicUpdateOp::getIndexInIndex());
            Tensor *in = producer->inTensor(DynamicUpdateOp::getInIndex());
            Tensor *outIdx =
                consumer->inTensor(DynamicSliceOp::getIndexInIndex());
            Tensor *out = consumer->outTensor(DynamicSliceOp::getOutIndex());
            // Check that both update+slice are based on the loop iterator
            if (inIdx->getGraph().removeScope(inIdx->id) ==
                    outIdx->getGraph().removeScope(outIdx->id) &&
                in->info == out->info &&
                dproducer->getAxes() == dconsumer->getAxes()) {
              resolved = true;
            } else {
              logging::transform::trace(
                  "[MergeLoops] Dynamic update {} / slice {} do not match "
                  "({}/{} {}/{} {}/{})",
                  producer->debugName(),
                  consumer->debugName(),
                  inIdx->getGraph().removeScope(inIdx->id),
                  outIdx->getGraph().removeScope(outIdx->id),
                  in->info,
                  out->info,
                  dproducer->getAxes(),
                  dconsumer->getAxes());
            }
          } else {
            logging::transform::trace(
                "[MergeLoops] Could not resolve path on {} {}",
                producer->debugName(),
                consumer->debugName());
          }
        }
      }

      if (!resolved) {
        logging::trace(
            "[MergeLoops] cannot merge {} -> {} (unresolved direct paths)",
            loopOp0->debugName(),
            loopOp1->debugName());
        merge = false;
        break;
      }

    } else if (t->hasProducer()) {
      // 1.) BFS for indirect paths
      std::queue<Tensor *> queue;
      std::set<Tensor *> visited;
      queue.push(t);
      while (!queue.empty()) {
        Tensor *tq = queue.front();
        queue.pop();
        if (tq->hasProducer()) {
          Op *tqp = tq->getProducer();
          if (LoopOp *tqlp = dynamic_cast<LoopOp *>(tqp)) {
            if (std::find(loopOps.begin(), loopOps.end(), tqlp) !=
                loopOps.end()) {
              // 1.) Indirect path, can't merge
              logging::trace(
                  "[MergeLoops] cannot merge {} -> {} (indirect path)",
                  loopOp0->debugName(),
                  tqlp->debugName());
              merge = false;
              queue = {};
              break;
            }
          }
          for (auto &idxAndTensor : tqp->input->tensorMap()) {
            if (visited.find(idxAndTensor.second) == visited.end()) {
              queue.push(idxAndTensor.second);
              visited.insert(tq);
            }
          }
        }
      }
      if (merge == false) {
        break;
      }
    }
  }

  return merge;
}

void MergeLoops::merge(const std::vector<LoopOp *> loops) const {
  logging::transform::trace("[MergeLoops] Processing {} loops.", loops.size());

  LoopOp *loop0 = loops.front();
  Graph &graph0 = loop0->getCalledGraph();
  Graph &graph  = loop0->getGraph();
  Ir &ir        = graph.getIr();

  for (size_t loopit = 1; loopit < loops.size(); ++loopit) {
    LoopOp *loop1 = loops.at(loopit);
    Graph &graph1 = loop1->getCalledGraph();

    logging::transform::trace("[MergeLoops] Merging loop {} into {}",
                              loop1->debugName(),
                              loop0->debugName());

    auto num_explicit0 = loop0->numExplicitInputs();
    // auto num_implicit0 = loop0->numImplicitInputs();
    auto num_explicit1 = loop1->numExplicitInputs();
    // auto num_implicit1 = loop1->numImplicitInputs();

    // Map from old to new subgraph tensor ID
    std::map<TensorId, TensorId> sgTensorRemap;

    // Check/move constants
    for (auto sg1ConstId : graph1.getTensors().getConstIds().v()) {
      auto sg0ConstId = graph0.addScope(graph1.removeScope(sg1ConstId));
      if (sg1ConstId.find(reservedConstValuePrefix()) != std::string::npos) {
        if (!graph0.getTensors().getConstIds().contains(sg0ConstId)) {
          Tensor *ct = graph1.getTensors().get(sg1ConstId);
          graph0.getTensors().addConstInit(
              sg0ConstId, ct->info, ct->tensorData()->data());
        }
        sgTensorRemap.insert({sg1ConstId, sg0ConstId});
      }
    }

    // Setup loop inputs in the following order:
    // 1. Explicit inputs of loop0 (existing)
    // 2. Explicit inputs of loop1 (newly added)
    // 3. Implicit inputs of loop0 (existing)
    // 4. Implicit inputs of loop1 (newly added)

    // 2. Add explicit inputs after the existing explicit inputs
    InIndex inOffset = num_explicit0;
    for (auto &input1 : loop1->input->tensorMap()) {
      if (input1.first >= num_explicit1) {
        // 4. Implicit inputs of loop1 at the end
        inOffset = loop0->input->maxIndex() + 1;
      }
      // Only process user-defined inputs
      if (input1.first >= LoopOp::getFirstInputInIndex()) {
        TensorId loop1sgInId =
            graph1.getInputId(loop1->opInToSubgraphInIndex(input1.first));

        // If the implicit input, or explicit input/output pair
        // is already connected to the loop, don't connect
        // a new input
        bool new_input = true;
        for (auto &input0 : loop0->input->tensorMap()) {
          if (input0.second->id == input1.second->id &&
              checkIdenticalPaths(
                  loop0, loop1, input0.first, input1.first, sgTensorRemap)) {
            new_input = false;
            TensorId loop0sgInId =
                graph0.getInputId(loop0->opInToSubgraphInIndex(input0.first));
            sgTensorRemap.insert({loop1sgInId, loop0sgInId});
          }
        }

        // If the input to loop1 is an output of loop0, don't connect a new
        // input, and connect internally instead
        for (auto &output0 : loop0->output->tensorMap()) {
          if (output0.second->id == input1.second->id) {
            new_input             = false;
            TensorId loop0sgOutId = graph0.getOutputId(
                loop0->opOutToSubgraphOutIndex(output0.first));
            sgTensorRemap.insert({loop1sgInId, loop0sgOutId});
          }
        }

        // Not connected yet, add input from loop1 onto loop0
        if (new_input) {
          TensorId loop0sgInId =
              graph0.addScope(graph1.removeScope(loop1sgInId));

          auto existingTensorIds =
              loop0->getCalledGraph().getTensors().getAllTensorIds();

          if (std::find(existingTensorIds.begin(),
                        existingTensorIds.end(),
                        loop0sgInId) != existingTensorIds.end()) {
            loop0sgInId = ir.createIntermediateTensorId(loop0sgInId);
          }

          sgTensorRemap.insert({loop1sgInId, loop0sgInId});

          logging::transform::trace("[MergeLoops] Adding input {} -> {}",
                                    input1.second->id,
                                    loop0sgInId);
          loop0->addLoopInput(
              inOffset++, input1.second->id, loop0sgInId, false);
        }
      }
    }

    // Move ops from loop1 into loop0
    for (Op *op : graph1.getOpSchedule({}, RequireOptimalSchedule::No)) {
      std::map<Op *, int, POpCmp> equivOps;

      for (auto &input : op->input->tensorMap()) {
        TensorId loop0sgId = sgTensorRemap.at(input.second->id);
        Tensor *t          = graph0.getTensors().get(loop0sgId);
        for (Op *consumer : t->consumers.getOps()) {
          if (consumer->getSubgraphEquivId() == op->getSubgraphEquivId() &&
              consumer->input->indices(t) == op->input->indices(input.second)) {
            equivOps[consumer]++;
          }
        }
      }

      bool new_op = true;
      for (auto equivOp : equivOps) {
        if (equivOp.second == op->input->n()) {
          // Equivalent Op found
          for (auto &output : op->output->tensorMap()) {
            TensorId loop1sgId = output.second->id;
            TensorId loop0sgId = equivOp.first->output->id(output.first);
            sgTensorRemap.insert({loop1sgId, loop0sgId});
          }
          logging::transform::trace("[MergeLoops] Skipping Op {}",
                                    op->debugName());
          new_op = false;
          break;
        }
      }

      if (new_op) {
        logging::transform::trace("[MergeLoops] Moving Op {} into {}",
                                  op->debugName(),
                                  loop0->debugName());
        auto clonedOpUp = op->clone();
        auto *clonedOp  = clonedOpUp.get();
        graph0.moveIntoGraph(std::move(clonedOpUp));
        for (auto &input : op->input->tensorMap()) {
          TensorId loop0sgId = sgTensorRemap.at(input.second->id);
          clonedOp->connectInTensor(input.first, loop0sgId);
        }
        for (auto &output : op->output->tensorMap()) {
          TensorId loop1sgId = output.second->id;
          TensorId loop0sgId = graph0.addScope(graph1.removeScope(loop1sgId));
          sgTensorRemap.insert({loop1sgId, loop0sgId});
          clonedOp->createAndConnectOutTensor(output.first, loop0sgId);
        }
        clonedOp->setup();
      }
    }

    // Add new outputs to loop0
    auto loop1OutMap = loop1->output->tensorMap();
    for (auto &output : loop1OutMap) {
      TensorId loop1sgId =
          graph1.getOutputId(loop1->opOutToSubgraphOutIndex(output.first));

      TensorId loop0sgId = sgTensorRemap.at(loop1sgId);
      auto sg0OutputIds  = graph0.getOutputIds();
      auto sgOut0It =
          std::find(sg0OutputIds.begin(), sg0OutputIds.end(), loop0sgId);

      loop1->disconnectOutTensor(output.second);

      if (sgOut0It == sg0OutputIds.end()) {
        // New subgraph (and Loop) output
        loop0->addLoopOutput(
            loop0->output->maxIndex() + 1, output.second->id, loop0sgId, false);
      } else {
        // Subgraph output already exists
        auto sgOut0Index = std::distance(sg0OutputIds.begin(), sgOut0It);
        auto out0Index   = loop0->subgraphOutToOpOutIndex(sgOut0Index);
        Tensor *out0     = loop0->output->tensor(out0Index);

        // Reconnect consumers
        for (Op *consumer : output.second->consumers.getOps()) {
          auto indices = consumer->input->indices(output.second);
          consumer->disconnectInTensor(output.second);
          for (InIndex index : indices) {
            consumer->connectInTensor(index, out0->id);
          }
        }

        auto &anchors = ir.getDataFlow().anchors();
        if (std::find(anchors.begin(), anchors.end(), output.second->id) !=
            anchors.end()) {
          // Identity Op to preserve both loop0 & loop1 output tensors
          auto identityOpUp = std::make_unique<IdentityOp>(
              Onnx::Operators::Identity_1, loop1->settings);
          Op *identityOp = identityOpUp.get();
          graph.moveIntoGraph(std::move(identityOpUp));
          identityOp->connectInTensor(IdentityOp::getInIndex(), out0->id);
          identityOp->connectOutTensor(IdentityOp::getOutIndex(),
                                       output.second->id);
        }
      }
    }

    logging::transform::trace("[MergeLoops] Erasing {} {} {}",
                              loop1->id,
                              loop1->debugName(),
                              loop1->getCalledGraph().id);

    if (graph.topoCons->contains(loop0, loop1)) {
      // TODO: Move these types of constraints into the Ops inside the loop body
      graph.topoCons->remove(loop0, loop1);
    }

    // graph.topoCons->transfer(loop1, loop0);
    loop1->disconnectAllInputs();
    loop1->disconnectAllOutputs();
    ir.removeGraph(loop1->getCalledGraph().id);
    graph.eraseOp(loop1->id);
  }

  loop0->setup();
  shortcutPaths(loop0);
  prunePaths(loop0);

  logging::transform::debug("[MergeLoops] Merging into loop {} {} done.",
                            loop0->debugName(),
                            loop0->getCalledGraph().id);

  if (logging::shouldLog(logging::Module::transform, logging::Level::Trace)) {
    logging::transform::trace("[MergeLoops] Loop subgraph schedule:");
    for (Op *op : graph0.getOpSchedule({}, RequireOptimalSchedule::No)) {
      logging::transform::trace(
          "[MergeLoops] {} - {}", op->getGraph().id.str(), op->debugName());
    }
  }
}

void MergeLoops::shortcutPaths(LoopOp *loop) const {
  logging::transform::debug("[MergeLoops] Applying shortcutPaths on {}",
                            loop->debugName());

  Graph &graph = loop->getCalledGraph();
  bool changed = true;
  while (changed) {
    changed = false;

    for (TensorId tid : graph.getTensors().getAllTensorIds()) {
      Tensor *t0 = graph.getTensors().get(tid);

      if (t0->hasProducer()) {
        Op *p0 = t0->getProducer();

        // Eliminate ReshapeOp -> ReshapeOp
        if (p0->isConvertibleTo<ReshapeOp>()) {
          Tensor *t1 = p0->input->tensor(ReshapeOp::getInIndex());
          if (t1->hasProducer()) {
            Op *p1 = t1->getProducer();
            if (p1->isConvertibleTo<ReshapeOp>()) {
              Tensor *t2 = p1->input->tensor(ReshapeOp::getInIndex());
              if (t2->info == t0->info) {
                changed = true;
                graph.replaceTensor(t0->id, t2->id);
                p0->disconnectAllInputs();
                p0->disconnectAllOutputs();
                graph.eraseOp(p0->id);
                logging::transform::trace(
                    "[MergeLoops] Shortcut {} -> {}", t0->id, t2->id);
              }
            }
          }
        } else

            // Eliminate DynamicUpdate -> DynamicSlice
            if (p0->isConvertibleTo<DynamicSliceOp>()) {
          Tensor *t1 = p0->input->tensor(DynamicSliceOp::getInIndex());
          Tensor *i1 = p0->input->tensor(DynamicSliceOp::getIndexInIndex());
          if (t1->hasProducer()) {
            Op *p1 = t1->getProducer();
            if (p1->isConvertibleTo<DynamicUpdateOp>()) {
              Tensor *t2 = p1->input->tensor(DynamicUpdateOp::getInIndex());
              Tensor *i2 =
                  p1->input->tensor(DynamicUpdateOp::getIndexInIndex());

              DynamicSliceOp *dp0  = dynamic_cast<DynamicSliceOp *>(p0);
              DynamicUpdateOp *dp1 = dynamic_cast<DynamicUpdateOp *>(p1);

              if (t2->info == t0->info && dp0->getAxes() == dp1->getAxes() &&
                  i1->id == i2->id) {
                changed = true;
                graph.replaceTensor(t0->id, t2->id);
                p0->disconnectAllInputs();
                p0->disconnectAllOutputs();
                graph.eraseOp(p0->id);
                logging::transform::trace(
                    "[MergeLoops] Shortcut {} -> {}", t0->id, t2->id);
              }
            }
          }
        }
      }
    }
  }
}

void MergeLoops::prunePaths(LoopOp *loop) const {
  logging::transform::debug("[MergeLoops] Applying prunePaths on {}",
                            loop->debugName());

  // Graph &graph = loop->getGraph();
  Graph &sgraph = loop->getCalledGraph();

  std::set<Op *> required;
  std::vector<Tensor *> front;

  using IoTuple = std::tuple<TensorId, TensorId, InIndex, OutIndex>;

  auto comp = [](const IoTuple &lhs, const IoTuple &rhs) {
    return std::get<2>(lhs) > std::get<3>(rhs);
  };

  // Subgraph input tensor -> Op input index, Op output index
  std::set<IoTuple, decltype(comp)> removablePairs(comp);

  for (auto &idxAndTensor : loop->output->tensorMap()) {
    OutIndex oidx    = idxAndTensor.first;
    InIndex iidx     = oidx + 2;
    TensorId sgOutId = sgraph.getOutputId(loop->opOutToSubgraphOutIndex(oidx));
    TensorId sgInId  = sgraph.getInputId(loop->opInToSubgraphInIndex(iidx));

    if (idxAndTensor.second->consumers.getOps().empty() &&
        !idxAndTensor.second->isGraphOutput() &&
        !idxAndTensor.second->isAnchored()) {
      logging::transform::trace(
          "[MergeLoops] Input/output pair prune candidate {} -> {}; {} -> {}",
          iidx,
          oidx,
          sgInId,
          sgOutId);
      removablePairs.insert({sgInId, sgOutId, iidx, oidx});
    } else {
      logging::transform::trace("[MergeLoops] Output {} required", sgOutId);
      // Output required -> output inside subgraph required
      front.push_back(sgraph.getTensors().get(sgOutId));
      front.push_back(sgraph.getTensors().get(sgInId));
    }
  }

  // Loop inputs 0: trip count & 1: termination condition can't be pruned
  front.push_back(sgraph.getTensors().get(sgraph.getInputId(
      loop->opInToSubgraphInIndex(LoopOp::getMaximumTripCountInIndex()))));
  front.push_back(sgraph.getTensors().get(sgraph.getInputId(
      loop->opInToSubgraphInIndex(LoopOp::getTerminationConditionInIndex()))));

  for (auto &opidAndOp : sgraph.getOps()) {
    if (opidAndOp.second->hasSideEffect()) {
      for (auto &tensor : opidAndOp.second->input->tensorMap()) {
        front.push_back(tensor.second);
      }
    }
    if (!opidAndOp.second->pruneable) {
      required.insert(opidAndOp.second.get());
    }
  }

  std::vector<Op *> opsToDelete;
  std::vector<Tensor *> tensorsToDelete;

  PruneHelper helper(&sgraph);

  bool changed = true;
  while (changed) {
    changed = false;

    helper.setFront(front);
    helper.setRequired(required);
    helper.analyze();

    opsToDelete     = helper.getOpsToDelete();
    tensorsToDelete = helper.getTensorsToDelete();

    for (auto it = removablePairs.begin(); it != removablePairs.end();) {
      auto removablePair = *it;
      TensorId sgInId    = std::get<0>(removablePair);
      TensorId sgOutId   = std::get<1>(removablePair);
      InIndex iidx       = std::get<2>(removablePair);
      OutIndex oidx      = std::get<3>(removablePair);

      Tensor *sgInTensor  = sgraph.getTensors().get(sgInId);
      Tensor *sgOutTensor = sgraph.getTensors().get(sgOutId);

      if (std::find(tensorsToDelete.begin(),
                    tensorsToDelete.end(),
                    sgInTensor) == tensorsToDelete.end()) {
        // After applying pruning, and if sgInTensor can't be removed,
        // the input/output pair of the loop can't be removed either.
        // Add the output tensor to the front and rerun pruning analysis.

        logging::transform::trace(
            "[MergeLoops] Cannot prune input/output pair {} -> {} ({} -> {})",
            iidx,
            oidx,
            sgInId,
            sgOutId);

        front.push_back(sgOutTensor);
        changed = true;
        it      = removablePairs.erase(it);
      } else {
        ++it;
      }
    }

    logging::transform::trace("[MergeLoops] Suggesting pruning {} ops, {} "
                              "tensors, changed: {}, removable pairs: {}",
                              opsToDelete.size(),
                              tensorsToDelete.size(),
                              changed,
                              removablePairs.size());
  }

  // Have to prune from high to low index, because input indices of
  // the loop change while pruning
  for (auto removablePair : removablePairs) {
    TensorId sgInId  = std::get<0>(removablePair);
    TensorId sgOutId = std::get<1>(removablePair);
    InIndex iidx     = std::get<2>(removablePair);
    OutIndex oidx    = std::get<3>(removablePair);
    logging::transform::trace(
        "[MergeLoops] Pruning input/output pair {} -> {} ({} -> {})",
        iidx,
        oidx,
        sgInId,
        sgOutId);
    loop->removeLoopInput(iidx);
    loop->removeLoopOutput(oidx);
  }

  helper.deleteOps(opsToDelete);
  helper.deleteTensors(tensorsToDelete);

  loop->setup();
}

bool MergeLoops::checkIdenticalPaths(
    LoopOp *loop0,
    LoopOp *loop1,
    InIndex opIn0,
    InIndex opIn1,
    std::map<TensorId, TensorId> sgTensorRemap) const {
  Tensor *opInT0 = loop0->input->tensor(opIn0);
  Tensor *opInT1 = loop1->input->tensor(opIn1);

  if (opInT0->id != opInT1->id) {
    // Not the same input
    return false;
  }

  if (opInT0->id == opInT1->id && opIn0 >= loop0->numExplicitInputs() &&
      opIn1 >= loop1->numExplicitInputs()) {
    // Identical implicit inputs, only one needs to be wired up
    return true;
  }

  // At this point, only explicit inputs should be left
  OutIndex opOut0 = opIn0 - 2;
  OutIndex opOut1 = opIn1 - 2;

  // Assume inputs are interchangeable
  sgTensorRemap.insert(
      {loop1->getCalledGraph().getInputId(loop1->opInToSubgraphInIndex(opIn1)),
       loop0->getCalledGraph().getInputId(
           loop0->opInToSubgraphInIndex(opIn0))});

  if (loop0->output->hasIndex(opOut0) && loop1->output->hasIndex(opOut1)) {
    std::queue<std::pair<Tensor *, Tensor *>> queue;
    queue.emplace(loop0->getCalledGraph().getTensors().get(
                      loop0->getCalledGraph().getOutputId(
                          loop0->opOutToSubgraphOutIndex(opOut0))),
                  loop1->getCalledGraph().getTensors().get(
                      loop1->getCalledGraph().getOutputId(
                          loop1->opOutToSubgraphOutIndex(opOut1))));
    while (!queue.empty()) {
      auto front = queue.front();
      queue.pop();
      if (front.first->hasProducer() && front.second->hasProducer()) {
        // Check if the tensor producer is equivalent between Loop0 and Loop1
        Op *op0 = front.first->getProducer();
        Op *op1 = front.second->getProducer();
        if (op0->getSubgraphEquivId() != op1->getSubgraphEquivId()) {
          return false;
        } else {
          auto m0 = op0->input->tensorMap();
          auto m1 = op1->input->tensorMap();
          for (auto idxAndTensor : m0) {
            queue.emplace(m0.at(idxAndTensor.first), m1.at(idxAndTensor.first));
          }
        }
      } else {
        // Check if the tensor was already remapped from Loop1 -> Loop0
        auto it = sgTensorRemap.find(front.second->id);
        if (it == sgTensorRemap.end() || it->second != front.first->id) {
          // Tensors not equivalent
          return false;
        }
      }
    }
  } else {
    return false;
  }

  return true;
}

std::size_t MergeLoops::id() { return typeid(MergeLoops).hash_code(); }

bool MergeLoops::apply(Graph &graph) const {
  auto schedule = graph.getOpSchedule({}, RequireOptimalSchedule::No);

  std::vector<std::vector<LoopOp *>> mergeSets;

  int64_t loopOpCount = 0;

  // Build sets of mergeable loops
  for (size_t i = 0; i < schedule.size(); ++i) {
    Op *op = schedule.at(i);
    if (LoopOp *loopOp0 = dynamic_cast<LoopOp *>(op)) {
      bool inserted = false;

      if (mergeSets.size() > 0) {
        auto &mergeSet = mergeSets.back();
        if (canMerge(mergeSet, loopOp0)) {
          inserted = true;
          mergeSet.push_back(loopOp0);
        }
      }
      if (!inserted) {
        mergeSets.push_back({loopOp0});
      }

      logging::transform::debug("[MergeLoops] position: {} processing loop {}, "
                                "mergeable with previous: {}",
                                i,
                                loopOp0->debugName(),
                                inserted);

      ++loopOpCount;
    } else {
      logging::transform::debug(
          "[MergeLoops] position: {} not a loop ({})", i, op->debugName());
    }
  }

  logging::transform::debug("[MergeLoops] Merging {} loops into {} loops",
                            loopOpCount,
                            mergeSets.size());

  // Merge loops
  for (auto &mergeSet : mergeSets) {
    if (mergeSet.size() > 1) {

      std::vector<std::string> loopOpNames;
      for (LoopOp *op : mergeSet) {
        loopOpNames.push_back(op->debugName());
      }

      logging::transform::trace(
          "[MergeLoops] Can merge: {} {}", loopOpNames.size(), loopOpNames);

      merge(mergeSet);
    }
  }

  return true;
}

namespace {
bool init = Transform::registerTransform(new MergeLoops);
}

} // namespace popart
