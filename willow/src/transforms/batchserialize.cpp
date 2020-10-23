#include <popart/error.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/names.hpp>
#include <popart/op.hpp>
#include <popart/op/concat.hpp>
#include <popart/op/dynamic/dynamicslice.hpp>
#include <popart/op/dynamic/dynamicupdate.hpp>
#include <popart/op/init.hpp>
#include <popart/op/iotilecopy.hpp>
#include <popart/op/ipucopy.hpp>
#include <popart/op/remote.hpp>
#include <popart/op/reshape.hpp>
#include <popart/op/slice.hpp>
#include <popart/tensor.hpp>
#include <popart/tensors.hpp>
#include <popart/topocons.hpp>
#include <popart/transforms/batchserialize.hpp>

namespace popart {

namespace {

TensorId createOrGetIndexTensor(Graph &graph, uint32_t index) {
  TensorId id = reservedIndexPrefix() + std::to_string(index);
  if (!graph.getTensors().contains(id)) {
    TensorInfo indexTensorInfo(DataType::UINT32, {1});
    std::vector<uint32_t> idData(1, index);
    graph.getTensors().addConstInit(
        id, indexTensorInfo, reinterpret_cast<void *>(idData.data()));
  }
  return id;
}

void connectOutTensor(Ir &ir, Op *op, TensorId id, OutIndex index) {
  if (ir.containsTensor(id)) {
    auto t = ir.getTensor(id);
    if (t->hasProducer()) {
      t->getProducer()->disconnectOutTensor(t);
    }
    op->connectOutTensor(index, id);
  } else {
    op->createAndConnectOutTensor(index, id);
  }
}

} // namespace

BatchSerialize::TraceFront::TraceFront(std::vector<Tensor *> tensors_,
                                       TraceDirection direction_)
    : tensors(tensors_), direction(direction_) {}

int64_t BatchSerialize::TraceFront::score() const {
  int64_t score = 0;
  if (direction == TraceDirection::Forward) {
    for (Tensor *t : tensors) {
      score += t->consumers.getOps().size();
    }
  } else {
    for (Tensor *t : tensors) {
      score += t->hasProducer() ? 1 : 0;
    }
  }
  return score;
}

// Trace fronts with less producers/consumers first
// (lesser chance of matching wrong isomorphic ops)
bool BatchSerialize::TraceFront::operator<(const TraceFront &rhs) const {
  return score() > rhs.score();
}

std::size_t BatchSerialize::id(int pass) {
  return typeid(BatchSerialize).hash_code() + pass;
}

OpId BatchSerialize::reshapeForSlice(Graph &graph,
                                     Op::Settings settings,
                                     TensorId inId,
                                     Shape newShape,
                                     TensorId outId,
                                     OptionalBatchSerializedPhase bsp) const {
  std::unique_ptr<ReshapeOp> reshapeOp = std::make_unique<ReshapeOp>(
      Onnx::AiOnnx::OpSet11::Reshape, newShape, settings);
  reshapeOp->setName("Batch_Reshape_" + inId);
  reshapeOp->setBatchSerializedPhase(bsp);
  Op *reshape = reshapeOp.get();
  graph.moveIntoGraph(std::move(reshapeOp));

  reshape->connectInTensor(ReshapeOp::getInIndex(), inId);
  connectOutTensor(graph.getIr(), reshape, outId, ReshapeOp::getOutIndex());
  reshape->setup();
  return reshape->id;
}

bool BatchSerialize::apply(Graph &graph) const {
  logging::transform::debug("[BatchSerialize] Started.");

  bool dynamicSlicing = true;
  bool dynamicConcat  = true;

  std::set<Op *> toErase;

  auto &ir               = graph.getIr();
  auto settings          = ir.getSessionOptions().batchSerializationSettings;
  int64_t batchSerFactor = settings.factor;
  std::map<std::pair<TensorId, TensorContext>, std::set<OpId>> batchSerialOps;

  auto getContext = [&](Op *op) {
    VGraphId vgid = op->hasVirtualGraphId() ? op->getVirtualGraphId() : -1;
    ExecutionPhase executionPhase =
        (ir.getSessionOptions().executionPhaseSettings.phases > 1 &&
         op->hasExecutionPhase())
            ? op->getExecutionPhase()
            : -1;
    PipelineStage pipelineStage =
        (ir.getSessionOptions().enablePipelining && op->hasPipelineStage())
            ? op->getPipelineStage()
            : -1;
    return TensorContext(
        settings.concatOnVirtualGraphChange ? vgid : unusedVGraphId,
        settings.concatOnExecutionPhaseChange ? executionPhase
                                              : unusedExecutionPhase,
        settings.concatOnPipelineStageChange ? pipelineStage
                                             : unusedPipelineStage);
  };

  // FWD
  if (pass == 1) {
    auto schedule = graph.getOpSchedule({}, RequireOptimalSchedule::No);

    std::set<TensorId> tensorsWithBatch;
    std::set<Op *> serializedOps;
    std::map<std::pair<TensorId, TensorContext>, std::vector<TensorId>>
        serializedTensorMap;
    std::map<std::pair<TensorId, TensorContext>, TensorId> concatTensorMap;

    for (TensorId id : ir.getTensorIds(TensorType::Stream)) {
      tensorsWithBatch.insert(id);
    }

    for (Op *op : schedule) {
      // Context in which the tensors are consumed
      TensorContext consumerContext = getContext(op);

      auto opInTensorIdxIds  = op->input->tensorIdMap();
      auto opOutTensorIdxIds = op->output->tensorIdMap();

      // TODO T20169: Improve: Pick up batch size/dimension from
      // previously serialized tensors.
      if (std::any_of(opInTensorIdxIds.begin(),
                      opInTensorIdxIds.end(),
                      [&tensorsWithBatch](
                          const std::pair<const InIndex, TensorId> &idxId) {
                        return tensorsWithBatch.find(idxId.second) !=
                               tensorsWithBatch.end();
                      })) {
        for (auto &outTensorIdxId : opOutTensorIdxIds) {
          if (op->getOutBatchAxis(outTensorIdxId.first) != -1) {
            tensorsWithBatch.insert(outTensorIdxId.second);
          }
        }
      }

      // Unsupported ops
      if (!op->canShard() || (op->toLoss == PathToLoss::Yes &&
                              op->fromLoss == PathFromLoss::Yes)) {
        logging::transform::trace("[BatchSerialize] Can not serialize {}",
                                  op->debugName());
        continue;
      } else {
        logging::transform::trace("[BatchSerialize] Serializing {}",
                                  op->debugName());
      }

      bool op_has_batch = false;
      for (auto &entry : op->input->indicesMap()) {
        auto type  = entry.first->getTensorTypeInfo()->type();
        auto shape = entry.first->info.shape();
        auto nelms = entry.first->info.nelms();

        // Check whether the producer is an init Op, if it has one
        bool isProducerInitOp = false;
        if (entry.first->hasProducer()) {
          isProducerInitOp =
              entry.first->getProducer()->isConvertibleTo<InitOp>();
        }

        auto producerContext = entry.first->hasProducer()
                                   ? getContext(entry.first->getProducer())
                                   : TensorContext(-1, -1, -1);

        auto serializedItProducer =
            serializedTensorMap.find({entry.first->id, producerContext});
        auto serializedItConsumer =
            serializedTensorMap.find({entry.first->id, consumerContext});

        logging::transform::trace(
            "[BatchSerialize] input tensor {} type: {} shape: {} "
            "serialized: [p: {} c: {}]",
            entry.first->id,
            type,
            shape,
            serializedItProducer != serializedTensorMap.end(),
            serializedItConsumer != serializedTensorMap.end());

        bool hasBatch =
            tensorsWithBatch.find(entry.first->id) != tensorsWithBatch.end() ||
            (isProducerInitOp && entry.first->getBatchAxis() != -1);
        // a.) Tensor can be serialized on the batch dimension
        // b.) Tensor has no producer, or is not yet registered in the
        // serialized tensor map
        if (hasBatch && (!entry.first->hasProducer() ||
                         serializedItProducer == serializedTensorMap.end() ||
                         serializedItConsumer == serializedTensorMap.end())) {

          // TODO T20169: Improve: Pick up batch size/dimension from
          // previously serialized tensors.
          // TODO T20169: Currently assuming all streams and actgrad
          // have batch dim

          op_has_batch |= nelms >= batchSerFactor;

          // TODO T20169: Support if batch dimension is not first.

          // c.) Tensor is not yet serialized in consumer context
          if (serializedItConsumer == serializedTensorMap.end()) {

            // Get the batch axis for this tensor
            int axis = entry.first->getBatchAxis();
            if (shape[axis] < batchSerFactor) {
              throw error("Batch axis: {} is smaller than the "
                          "batch serialisation factor: {} for tensor {}",
                          shape[axis],
                          batchSerFactor,
                          entry.first->id);
            }
            logging::transform::trace(
                "[BatchSerialize] batch axis for {} is {}",
                entry.first->id,
                axis);
            int batch_slice_size =
                static_cast<int>(shape[axis] / batchSerFactor);

            TensorId sliceableTensorId;

            // Reshape to minimize sliceable offsets along the axis dimension
            if (batch_slice_size > 1) {

              Shape reshape(shape.size() + 1);
              for (size_t i = 0; i < reshape.size(); ++i) {
                if (i < axis) {
                  reshape[i] = shape[i];
                } else if (i > axis + 1) {
                  reshape[i] = shape[i - 1];
                } else if (i == axis + 1) {
                  reshape[i] = batch_slice_size;
                } else if (i == axis) {
                  reshape[i] = batchSerFactor;
                }
              }

              logging::transform::trace(
                  "[BatchSerialize] Reshape to sliceable: [{} -> {}]",
                  shape,
                  reshape);

              sliceableTensorId =
                  ir.createIntermediateTensorId(entry.first->id);
              batchSerialOps[{entry.first->id, consumerContext}].insert(
                  reshapeForSlice(graph,
                                  op->getSettings(),
                                  entry.first->id,
                                  reshape,
                                  sliceableTensorId,
                                  OptionalBatchSerializedPhase()));

            } else {
              sliceableTensorId = entry.first->id;
            }

            for (int64_t b = 0; b < batchSerFactor; ++b) {

              Op *slice = nullptr;
              if (dynamicSlicing) {
                std::vector<int64_t> axesv(1, axis);
                std::vector<int64_t> sizesv(1, 1);

                std::unique_ptr<DynamicSliceOp> sliceOp =
                    std::make_unique<DynamicSliceOp>(
                        Onnx::CustomOperators::DynamicSlice_1,
                        axesv,
                        sizesv,
                        true,
                        op->getSettings());
                sliceOp->setName("BatchSlice_" + entry.first->id);
                slice = sliceOp.get();
                graph.moveIntoGraph(std::move(sliceOp));
                batchSerialOps[{entry.first->id, consumerContext}].insert(
                    slice->id);
                slice->setBatchSerializedPhase(b);
              } else {
                std::vector<int64_t> startsv(1, b);
                // TODO T20169: Factor support
                std::vector<int64_t> endsv(1, (b + 1));
                // TODO T20169: Different axis support
                std::vector<int64_t> axesv(1, axis);

                std::unique_ptr<SliceOp> sliceOp =
                    std::make_unique<SliceOp>(Onnx::AiOnnx::OpSet11::Slice,
                                              startsv,
                                              endsv,
                                              axesv,
                                              std::vector<int64_t>{}, // steps
                                              op->getSettings());
                sliceOp->setName("BatchSlice_" + entry.first->id);
                slice = sliceOp.get();
                graph.moveIntoGraph(std::move(sliceOp));
                batchSerialOps[{entry.first->id, consumerContext}].insert(
                    slice->id);
                slice->setBatchSerializedPhase(-1);
              }
              // Slice should always happen on the consumer side.
              if (std::get<0>(consumerContext) > -1)
                slice->setVirtualGraphId(std::get<0>(consumerContext));
              if (std::get<1>(consumerContext) > -1)
                slice->setExecutionPhase(std::get<1>(consumerContext));
              if (std::get<2>(consumerContext) > -1)
                slice->setPipelineStage(std::get<2>(consumerContext));
              slice->connectInTensor(SliceOp::getInIndex(), sliceableTensorId);
              if (dynamicSlicing) {
                slice->connectInTensor(
                    DynamicSliceOp::getIndexInIndex(),
                    createOrGetIndexTensor(graph, static_cast<uint32_t>(b)));
              }
              TensorId sliceId =
                  ir.createBatchSliceTensorId(entry.first->id,
                                              static_cast<unsigned int>(b),
                                              static_cast<unsigned int>(b + 1));
              slice->createAndConnectOutTensor(SliceOp::getOutIndex(), sliceId);
              slice->setup();

              logging::transform::trace("Slice tensor {} {} -> {} {}",
                                        entry.first->id,
                                        entry.first->info.shape(),
                                        sliceId,
                                        ir.getTensor(sliceId)->info.shape());

              if (dynamicSlicing && batch_slice_size > 1) {

                Shape reshape(shape.size());
                for (size_t i = 0; i < shape.size(); ++i) {
                  if (i != axis) {
                    reshape[i] = shape[i];
                  } else {
                    reshape[i] = batch_slice_size;
                  }
                }

                logging::transform::trace(
                    "[BatchSerialize] Reshape slice: [{} -> {}]",
                    ir.getTensor(sliceId)->info.shape(),
                    reshape);

                TensorId sliceReshapedId =
                    ir.createIntermediateTensorId(entry.first->id);
                batchSerialOps[{entry.first->id, consumerContext}].insert(
                    reshapeForSlice(graph,
                                    op->getSettings(),
                                    sliceId,
                                    reshape,
                                    sliceReshapedId,
                                    b));

                serializedTensorMap[{entry.first->id, consumerContext}]
                    .push_back(sliceReshapedId);
              } else {
                serializedTensorMap[{entry.first->id, consumerContext}]
                    .push_back(sliceId);
              }
            }
            if (consumerContext == producerContext) {
              concatTensorMap[{entry.first->id, producerContext}] =
                  entry.first->id;
            }
          }
        } else if (serializedItProducer != serializedTensorMap.end() ||
                   serializedItConsumer != serializedTensorMap.end()) {
          // Input already serialized
          op_has_batch |= true;
        }
      }

      // Operations not affected by the batch size can skip this part
      if (op_has_batch) {
        std::map<TensorId, std::vector<TensorId>> shardInputs;

        for (auto &in : op->input->tensorMap()) {
          auto serializedTensor =
              serializedTensorMap.find({in.second->id, consumerContext});
          if (serializedTensor != serializedTensorMap.end()) {
            // Tensors split along batch dimension
            for (int64_t b = 0; b < batchSerFactor; ++b) {
              shardInputs[in.second->id].push_back(serializedTensor->second[b]);
            }
          }
        }

        std::map<TensorId, std::vector<TensorId>> shardOutputs;
        // The following will throw an error if batch serialisation failed
        // to slice a tensor. Return a sensible error message.
        try {
          shardOutputs = op->shard(shardInputs);
        } catch (error &e) {
          std::stringstream ss;
          ss << "Batch serialisation failed while processing op " << op->opid;
          ss << ". The inputs to this op are: ";
          for (unsigned j = 0; j < op->inTensorCount(); j++) {
            ss << op->inId(j) << ((j < op->inTensorCount() - 1) ? ", " : ".");
          }
          throw error(ss.str());
        }

        for (auto &idkv : shardOutputs) {
          if (idkv.second.size() == batchSerFactor) {
            serializedTensorMap[{idkv.first, consumerContext}] = idkv.second;
          }
        }

        toErase.insert(op);
      }
    }

    // Make sure nobody consumes the original tensors of a serialized tensor.
    // If there are still consumers, concat the slices and reconnect.
    for (auto serializedTensor : serializedTensorMap) {
      Tensor *tensor = graph.getTensors().get(serializedTensor.first.first);

      if (!tensor->hasProducer())
        continue;

      Op *producer         = tensor->getProducer();
      auto producerContext = getContext(producer);
      if (serializedTensor.first.second != producerContext) {
        continue;
      }

      auto concatIfNecessary = [this,
                                &tensor,
                                &producer,
                                &ir,
                                &graph,
                                &concatTensorMap,
                                &producerContext,
                                &serializedTensor,
                                &batchSerialOps,
                                &dynamicConcat,
                                &batchSerFactor]() {
        if (concatTensorMap.find({tensor->id, producerContext}) ==
            concatTensorMap.end()) {
          // TODO T20169: Different axis support
          TensorId serId0 = serializedTensor.second[0];
          Tensor *serT0   = ir.getTensor(serId0);
          int64_t axis    = 0;
          for (unsigned i = 0; i < tensor->info.shape().size(); ++i) {
            if (serT0->info.shape()[i] < tensor->info.shape()[i]) {
              axis = i;
              break;
            }
          }

          TensorId concatId = serializedTensor.first.first;

          if (dynamicConcat) {
            Tensor *t = graph.getTensors().get(serializedTensor.first.first);

            TensorId lastId;
            for (size_t b = 0; b < serializedTensor.second.size(); ++b) {
              TensorId sliceTensorId = serializedTensor.second[b];
              Tensor *s              = graph.getTensors().get(sliceTensorId);

              logging::transform::trace(
                  "[BatchSerialize] Concat slice {} ({}) of {} ({})",
                  s->id,
                  s->info.shape(),
                  t->id,
                  t->info.shape());

              size_t batch_slice_size = s->info.shape()[axis];

              auto outShape   = t->info.shape();
              auto initShape  = t->info.shape();
              auto sliceShape = s->info.shape();

              TensorId toUpdateSliceTensorId;
              if (dynamicConcat && batch_slice_size > 1) {

                initShape.resize(initShape.size() + 1);
                sliceShape.resize(sliceShape.size() + 1);
                for (size_t i = 0; i < initShape.size(); ++i) {
                  if (i < axis) {
                    initShape[i]  = outShape[i];
                    sliceShape[i] = outShape[i];
                  } else if (i > axis + 1) {
                    initShape[i]  = outShape[i - 1];
                    sliceShape[i] = outShape[i - 1];
                  } else if (i == axis + 1) {
                    initShape[i]  = batch_slice_size;
                    sliceShape[i] = batch_slice_size;
                  } else if (i == axis) {
                    initShape[i]  = batchSerFactor;
                    sliceShape[i] = 1;
                  }
                }

                logging::transform::trace(
                    "[BatchSerialize] Reshape for update: [{} -> {}, {}]",
                    outShape,
                    initShape,
                    sliceShape);

                toUpdateSliceTensorId =
                    ir.createIntermediateTensorId(sliceTensorId);
                batchSerialOps[{tensor->id, producerContext}].insert(
                    reshapeForSlice(graph,
                                    producer->getSettings(),
                                    sliceTensorId,
                                    sliceShape,
                                    toUpdateSliceTensorId,
                                    b));
              } else {
                toUpdateSliceTensorId = sliceTensorId;
              }

              if (b == 0) {

                TensorInfo info = t->info;
                info.set(info.dataType(), initShape);

                auto initOp =
                    std::make_unique<InitOp>(Onnx::CustomOperators::Init_1,
                                             info,
                                             TensorType::ActGrad,
                                             InitType::Zero,
                                             producer->getSettings());
                Op *init = initOp.get();
                init->setName("ConcatInit_" + concatId);
                init->setBatchSerializedPhase(OptionalBatchSerializedPhase());
                graph.moveIntoGraph(std::move(initOp));
                batchSerialOps[{tensor->id, producerContext}].insert(init->id);
                lastId = ir.createIntermediateTensorId(concatId);
                init->createAndConnectOutTensor(InitOp::getOutIndex(), lastId);
                init->setup();
              }

              std::vector<int64_t> axesv(1, axis);
              std::vector<int64_t> sizesv(1, 1);

              std::unique_ptr<DynamicUpdateOp> updateOp =
                  std::make_unique<DynamicUpdateOp>(
                      Onnx::CustomOperators::DynamicUpdate_1,
                      axesv,
                      sizesv,
                      true,
                      producer->getSettings());
              updateOp->setName("BatchConcat_" + concatId);
              DynamicUpdateOp *update = updateOp.get();
              graph.moveIntoGraph(std::move(updateOp));
              batchSerialOps[{tensor->id, producerContext}].insert(update->id);
              update->setBatchSerializedPhase(b);

              if (std::get<0>(producerContext) > -1)
                update->setVirtualGraphId(std::get<0>(producerContext));
              if (std::get<1>(producerContext) > -1)
                update->setExecutionPhase(std::get<1>(producerContext));
              if (std::get<2>(producerContext) > -1)
                update->setPipelineStage(std::get<2>(producerContext));

              update->connectInTensor(DynamicUpdateOp::getInIndex(),
                                      toUpdateSliceTensorId);
              update->connectInTensor(
                  DynamicUpdateOp::getIndexInIndex(),
                  createOrGetIndexTensor(graph, static_cast<uint32_t>(b)));
              update->connectInTensor(DynamicUpdateOp::getUpdateInIndex(),
                                      lastId);

              update->settings.inferTensorMappingToFrom.insert(
                  {DynamicUpdateOp::getUpdateInIndex(),
                   DynamicUpdateOp::getInIndex()});

              lastId = (b == serializedTensor.second.size() - 1 &&
                        batch_slice_size == 1)
                           ? concatId
                           : ir.createIntermediateTensorId(concatId);
              connectOutTensor(
                  ir, update, lastId, DynamicUpdateOp::getOutIndex());
              update->setup();

              if (b == serializedTensor.second.size() - 1 &&
                  batch_slice_size > 1 && dynamicConcat) {

                logging::transform::trace(
                    "[BatchSerialize] Reshape after last update: [{} -> {}]",
                    initShape,
                    outShape);

                batchSerialOps[{tensor->id, producerContext}].insert(
                    reshapeForSlice(graph,
                                    producer->getSettings(),
                                    lastId,
                                    outShape,
                                    concatId,
                                    OptionalBatchSerializedPhase()));
              }
            }
          } else {
            std::unique_ptr<ConcatOp> concatOp = std::make_unique<ConcatOp>(
                Onnx::AiOnnx::OpSet11::Concat, axis, producer->getSettings());
            concatOp->setName("BatchConcat_" + concatId);
            // Concat should always happen on the producer side.
            ConcatOp *concat = concatOp.get();
            graph.moveIntoGraph(std::move(concatOp));
            batchSerialOps[{tensor->id, producerContext}].insert(concat->id);
            if (std::get<0>(producerContext) > -1)
              concat->setVirtualGraphId(std::get<0>(producerContext));
            if (std::get<1>(producerContext) > -1)
              concat->setExecutionPhase(std::get<1>(producerContext));
            if (std::get<2>(producerContext) > -1)
              concat->setPipelineStage(std::get<2>(producerContext));
            for (size_t b = 0; b < serializedTensor.second.size(); ++b) {
              concat->connectInTensor(static_cast<popart::InIndex>(b),
                                      serializedTensor.second[b]);
            }
            concat->createAndConnectOutTensor(ConcatOp::getOutIndex(),
                                              concatId);
            concat->setup();
          }

          concatTensorMap[{serializedTensor.first.first, producerContext}] =
              concatId;
        }
      };

      // Anchors that need the concatenated tensor
      auto &anchors = ir.getDataFlow().anchors();
      if (std::find(anchors.begin(), anchors.end(), tensor->id) !=
          anchors.end()) {
        concatIfNecessary();
      }

      // Consumers that need the concatenated tensor
      for (auto consumer : tensor->consumers.getOps()) {

        // Not important what OPs that are going to be removed are consuming
        if (toErase.find(consumer) != toErase.end())
          continue;

        auto batchSerialOpsForTensor =
            batchSerialOps.find(serializedTensor.first);
        if (batchSerialOpsForTensor != batchSerialOps.end()) {
          // Consumers involved in producing the serialized tensor are exempt
          if (batchSerialOpsForTensor->second.find(consumer->id) !=
              batchSerialOpsForTensor->second.end()) {
            continue;
          }
        }

        logging::transform::trace(
            "[BatchSerialize] Consumer {} is still consuming {}.",
            consumer->debugName(),
            tensor->id);

        auto indices = consumer->input->indices(tensor);

        concatIfNecessary();

        // Add concatenated tensor
        for (auto i : indices) {
          consumer->disconnectInTensor(i, tensor);
          consumer->connectInTensor(
              i, concatTensorMap.at({tensor->id, producerContext}));
        }
      }
    }

    // Remove all ops that have been serialized
    for (Op *op : toErase) {
      logging::trace("[BatchSerialize] Erasing op {}", op->debugName());
      op->disconnectAllInputs();
      op->disconnectAllOutputs();
      graph.eraseOp(op->id);
    }
  }

  // Annotate priorities to isolate batch ops and crystallize the schedule
  // between batch serial phases
  if (pass == 2) {
    auto schedule = graph.getOpSchedule({}, RequireOptimalSchedule::Yes);

    // If batchSchedule == Scheduler we defer any further scheduling to the
    // scheduler.
    if (settings.batchSchedule != BatchSerializationBatchSchedule::Scheduler) {

      // Local isomorphism distinction gap
      int64_t scoreGap = settings.isomorphismScoreGap;

      // Crystallize schedule within batch serialized phase by inserting topo
      // cons
      std::map<Op *, int64_t> opScheduleIndex;
      std::map<Op *, SubgraphEquivId> opSubgraphEquivId;

      for (size_t i = 0; i < schedule.size(); ++i) {
        opScheduleIndex[schedule.at(i)] = i;
        opSubgraphEquivId[schedule.at(i)] =
            schedule.at(i)->getSubgraphEquivId();
      }

      std::set<Op *> equivProcessedOps;
      OpToSection opToSection;
      OpToPosition opToPosition;
      PositionsToOp positionToOp;
      OpsBehindSection opsBehindSection;
      IsoScoreCache cachedIsoScores;

      // Find equivalence classes, derive positions
      Section section   = -1;
      Position position = 0;
      bool nextSection  = true;
      for (Op *op : schedule) {
        logging::transform::trace(
            "[BatchSerialize] BSP: {} S: {} P: {} prio: {} OP: {}, SID: {}",
            op->hasBatchSerializedPhase()
                ? std::to_string(op->getBatchSerializedPhase())
                : "*",
            section,
            position,
            op->settings.schedulePriority,
            op->debugName(),
            op->getSubgraphEquivId());
        if (op->hasBatchSerializedPhase()) {
          auto bsp = op->getBatchSerializedPhase();
          if (bsp == 0) {
            if (nextSection) {
              ++section;
              nextSection = false;
            }
            opToPosition[{section, bsp}][op]       = position;
            positionToOp[{section, bsp}][position] = op;
            opToSection[op]                        = section;

            // First batch defines schedule order
            position++;
          } else if (bsp > 0) {
            nextSection = true;
          }
        } else {
          // Ops with no annotated bsp that occur after a section
          opsBehindSection[section].push_back(op);
        }
      }

      // Find seed fronts
      auto parallelTraceFront =
          findParallelTraceFronts(schedule, batchSerFactor);

      logging::trace("[BatchSerialize] Parallel trace fronts: {}",
                     parallelTraceFront.size());

      std::set<std::pair<std::vector<Tensor *>, TraceDirection>> visited;

      while (!parallelTraceFront.empty()) {
        std::map<std::tuple<OpId, TraceDirection, int>, std::vector<Tensor *>>
            nextFronts;
        auto traceFront = parallelTraceFront.top();
        auto &tensors   = traceFront.tensors;
        auto &direction = traceFront.direction;
        parallelTraceFront.pop();

        std::vector<TensorId> ids;
        for (Tensor *t : tensors) {
          ids.push_back(t->id);
        }

        logging::transform::trace(
            "[BatchSerialize] Current ({}) front: {} (score: {}, remaining: "
            "{})",
            direction == TraceDirection::Forward ? "forward" : "backward",
            ids,
            traceFront.score(),
            parallelTraceFront.size());

        std::vector<Tensor *> frontTensors;
        std::vector<std::vector<Op *>> frontOps;
        visited.insert({tensors, direction});
        for (Tensor *t : tensors) {
          std::vector<Op *> fops;
          if (direction == TraceDirection::Forward) {
            fops = t->consumers.getOps();
            frontOps.push_back(fops);
          } else {
            if (t->hasProducer()) {
              fops.push_back(t->getProducer());
              frontOps.push_back(fops);
            } else {
              // Change direction on tensors without producers
              frontTensors.push_back(t);
            }
          }
        }
        if (frontTensors.size() != 0) {
          nextFronts[{-1, TraceDirection::Forward, -1}] = frontTensors;
        }

        // Skip tracing of certain tensors that can lead to false
        // positive isomporphism results
        if (std::any_of(tensors.begin(), tensors.end(), [](Tensor *t) {
              TensorId id = t->id;
              return id.find(reservedIndexPrefix()) != std::string::npos ||
                     id.find(reservedRandomSeedPrefix()) != std::string::npos;
              ;
            })) {
          continue;
        }

        if (!frontOps.empty() && !std::any_of(frontOps.begin(),
                                              frontOps.end(),
                                              [](const std::vector<Op *> &ops) {
                                                return ops.empty();
                                              })) {
          for (Op *op0 : frontOps.front()) {
            if (!op0->hasBatchSerializedPhase() ||
                op0->getBatchSerializedPhase() != 0 ||
                equivProcessedOps.find(op0) != equivProcessedOps.end()) {
              continue;
            }
            equivProcessedOps.insert(op0);

            section = opToSection.at(op0);
            std::vector<Op *> equivalentOps;
            std::set<BatchSerializedPhase> foundBSPs;
            foundBSPs.insert(op0->getBatchSerializedPhase());

            for (auto tensorAndIndex : op0->output->indicesMap()) {
              for (InIndex index : tensorAndIndex.second) {
                nextFronts[{op0->id, TraceDirection::Forward, index}].push_back(
                    tensorAndIndex.first);
              }
            }
            for (auto tensorAndIndex : op0->input->indicesMap()) {
              for (InIndex index : tensorAndIndex.second) {
                nextFronts[{op0->id, TraceDirection::Backward, index}]
                    .push_back(tensorAndIndex.first);
              }
            }

            std::map<BatchSerializedPhase, std::vector<Op *>> binnedOps;

            for (auto ops : frontOps) {
              // Iterate through potentially isomorphic ops
              for (Op *op1 : ops) {
                if (op1->id != op0->id && op1->toLoss == op0->toLoss &&
                    op1->fromLoss == op0->fromLoss &&
                    opSubgraphEquivId[op1] == opSubgraphEquivId[op0] &&
                    op1->hasBatchSerializedPhase() &&
                    foundBSPs.find(op1->getBatchSerializedPhase()) ==
                        foundBSPs.end() &&
                    equivProcessedOps.find(op1) == equivProcessedOps.end()) {
                  binnedOps[op1->getBatchSerializedPhase()].push_back(op1);
                }
              }
            }

            for (auto &phaseAndOps : binnedOps) {
              // Pick the top Op from each batch serialized phase bin
              auto &binOps = phaseAndOps.second;
              // Get element with highest local isomorphism score against op0
              Op *op1 = *std::max_element(
                  binOps.begin(),
                  binOps.end(),
                  [this,
                   &cachedIsoScores,
                   &opToSection,
                   &opToPosition,
                   &opSubgraphEquivId,
                   &op0,
                   &scoreGap](Op *lhs, Op *rhs) {
                    std::set<std::pair<Op *, Op *>> visitedOpsLhs;
                    std::set<std::pair<Op *, Op *>> visitedOpsRhs;
                    if (lhs->id == rhs->id) {
                      return false;
                    }
                    int depth            = 1;
                    int64_t lhsScore     = 0;
                    int64_t rhsScore     = 0;
                    int64_t lastLhsScore = 0;
                    int64_t lastRhsScore = 0;
                    do {
                      visitedOpsLhs.clear();
                      visitedOpsRhs.clear();
                      lastLhsScore = lhsScore;
                      lastRhsScore = rhsScore;
                      lhsScore     = getLocalIsoScore(cachedIsoScores,
                                                  opToSection,
                                                  opToPosition,
                                                  opSubgraphEquivId,
                                                  {op0, lhs},
                                                  visitedOpsLhs,
                                                  depth,
                                                  true);
                      rhsScore     = getLocalIsoScore(cachedIsoScores,
                                                  opToSection,
                                                  opToPosition,
                                                  opSubgraphEquivId,
                                                  {op0, rhs},
                                                  visitedOpsRhs,
                                                  depth,
                                                  true);
                      ++depth;
                    } while (std::abs(lhsScore - rhsScore) < scoreGap &&
                             lastLhsScore != lhsScore &&
                             lastRhsScore != rhsScore);
                    return lhsScore < rhsScore;
                  });

              BatchSerializedPhase bsp = op1->getBatchSerializedPhase();
              foundBSPs.insert(bsp);

              for (auto tensorAndIndex : op1->output->indicesMap()) {
                for (InIndex index : tensorAndIndex.second) {
                  nextFronts[{op0->id, TraceDirection::Forward, index}]
                      .push_back(tensorAndIndex.first);
                }
              }
              for (auto tensorAndIndex : op1->input->indicesMap()) {
                for (InIndex index : tensorAndIndex.second) {
                  nextFronts[{op0->id, TraceDirection::Backward, index}]
                      .push_back(tensorAndIndex.first);
                }
              }

              auto pos = opToPosition[{section, 0}][op0];
              opToPosition[{section, bsp}][op1] = pos;
              positionToOp[{section, bsp}][pos] = op1;
              opToSection[op1]                  = section;
              equivProcessedOps.insert(op1);
            }
          }
        }
        for (auto nextFront : nextFronts) {
          bool alreadyVisited =
              visited.find({nextFront.second, std::get<1>(nextFront.first)}) !=
              visited.end();
          if (alreadyVisited || nextFront.second.size() != batchSerFactor) {
            std::vector<TensorId> idsLocal;
            for (Tensor *tx : nextFront.second) {
              idsLocal.push_back(tx->id);
            }
            logging::transform::trace(
                "[BatchSerialize] Front {}{} size {} is a deadend",
                idsLocal,
                alreadyVisited ? " (already visited)" : "",
                idsLocal.size());
          } else {
            // All front tensors for the different BSPs have been found
            parallelTraceFront.push(
                TraceFront(nextFront.second, std::get<1>(nextFront.first)));
          }
        }
      }

      // Check if for any BSP > 0, the isomorphic Op in BSP 0 could not be found
      // and clean up BSP settings on non-isomorphic Ops
      section = -1;
      for (Op *op : schedule) {
        if (op->hasBatchSerializedPhase() &&
            op->getBatchSerializedPhase() >= 0) {
          if (opToSection.find(op) == opToSection.end()) {
            logging::warn("[BatchSerialize] Could not find isomorphic "
                          "position for {}",
                          op->debugName());
            op->setBatchSerializedPhase(OptionalBatchSerializedPhase());
            opsBehindSection[section].push_back(op);
          } else {
            section   = opToSection.at(op);
            auto bsp0 = op->getBatchSerializedPhase();
            if (bsp0 == 0) {
              auto pos       = opToPosition.at({section, bsp0}).at(op);
              bool hasIsoOps = false;
              // Check if the Op with BSP == 0 has any isomorphic operations
              for (auto bsp1 = 1; bsp1 < batchSerFactor; ++bsp1) {
                auto posToOp = positionToOp.find({section, bsp1});
                hasIsoOps |=
                    (posToOp != positionToOp.end() &&
                     posToOp->second.find(pos) != posToOp->second.end());
              }
              if (!hasIsoOps) {
                logging::warn("[BatchSerialize] Could not find isomorphic "
                              "position for {}",
                              op->debugName());
                opToSection.erase(op);
                op->setBatchSerializedPhase(OptionalBatchSerializedPhase());
              }
            }
          }
        }
      }

      const bool isOverlapSchedule =
          (settings.batchSchedule ==
           BatchSerializationBatchSchedule::OverlapOnIo) ||
          (settings.batchSchedule ==
           BatchSerializationBatchSchedule::OverlapOnCompute);

      if (isOverlapSchedule) {
        // Swap positions of operations inside the bins
        tryToMakeAmenableToParallelization(graph, positionToOp);
      }

      // Crystallize schedule within each batch serialized phase
      for (auto &positions : positionToOp) {
        Op *prev = nullptr;
        for (auto &pos : positions.second) {
          logging::transform::trace("[BatchSerialize] Fixed: {} {} {} {}",
                                    positions.first.first,
                                    positions.first.second,
                                    pos.first,
                                    pos.second->debugName());
          Op *op = pos.second;
          if (prev) {
            graph.topoCons->insert(prev, op, true);
          }
          prev = op;
        }
        if (prev) {
          for (Op *op : opsBehindSection[positions.first.first]) {
            // Stop Ops behind the batch serialized phases to slide up
            // in the schedule
            graph.topoCons->insert(prev, op);
          }
        }
      }

      if (isOverlapSchedule) {
        // Insert constraints to interleave loading/computing
        // between batch serialized phases
        addParallelizationConstraints(graph, positionToOp);
      }
    }

    // Freeze the schedule completely, so that the batch serial order cannot be
    // undone by outlining
    graph.freezeSchedule({});
  }

  logging::transform::debug("[BatchSerialize] Done.");
  return true;
}

std::priority_queue<BatchSerialize::TraceFront>
BatchSerialize::findParallelTraceFronts(std::vector<Op *> schedule,
                                        int64_t batchSerFactor) const {
  std::priority_queue<BatchSerialize::TraceFront> queue;
  std::vector<Tensor *> traceTensors(batchSerFactor);

  for (Op *op : schedule) {

    if (!op->hasBatchSerializedPhase()) {

      // Breadth first search, find patterns like:
      // 1.)
      //
      // |           |           |
      // Op (BSP 0)  Op (BSP 1)  Op (BSP 2)
      //    \        |         /
      //     '------ Op ------'
      //             |
      //
      // 2.)
      //
      // |           |           |
      // Op (BSP 0)  |           |
      // '---------- Op (BSP 1)  |
      //             '---------- Op (BSP 2)
      //                         '------------ Op

      std::map<SubgraphEquivId, std::vector<Tensor *>> equivTraceTensors;

      std::set<BatchSerializedPhase> foundBSPs;
      std::set<Op *> visited;
      std::queue<Op *> opQueue;
      opQueue.push(op);
      while (!opQueue.empty() && foundBSPs.size() != batchSerFactor) {
        Op *op0 = opQueue.front();
        opQueue.pop();
        visited.insert(op0);
        for (auto &idxAndTensor : op0->input->tensorMap()) {
          if (idxAndTensor.second->hasProducer()) {
            Op *op1 = idxAndTensor.second->getProducer();
            if (op1->hasBatchSerializedPhase()) {
              BatchSerializedPhase bsp1 = op1->getBatchSerializedPhase();
              SubgraphEquivId id        = op1->getSubgraphEquivId();
              if (equivTraceTensors.find(id) == equivTraceTensors.end()) {
                equivTraceTensors.insert(
                    {id, std::vector<Tensor *>(batchSerFactor)});
              }
              if (!equivTraceTensors.at(id).at(bsp1)) {
                equivTraceTensors.at(id)[bsp1] = idxAndTensor.second;
              }
              if (std::all_of(
                      equivTraceTensors.at(id).begin(),
                      equivTraceTensors.at(id).end(),
                      [](const Tensor *t) { return static_cast<bool>(t); })) {
                traceTensors = equivTraceTensors.at(id);
                opQueue      = {};
                break;
              }
            }
            opQueue.push(op1);
          }
        }
      }

      if (std::all_of(traceTensors.begin(),
                      traceTensors.end(),
                      [](const Tensor *t) { return static_cast<bool>(t); })) {
        queue.push(BatchSerialize::TraceFront(
            traceTensors, BatchSerialize::TraceDirection::Backward));
      }
    }
  }
  return queue;
}

int64_t BatchSerialize::getLocalIsoScore(
    IsoScoreCache &cachedIsoScores,
    const OpToSection &opToSection,
    const OpToPosition &opToPosition,
    std::map<Op *, SubgraphEquivId> &opSubgraphEquivId,
    std::pair<Op *, Op *> ops,
    std::set<std::pair<Op *, Op *>> &visitedOps,
    int maxDepth,
    bool cached) const {
  if (cached) {
    auto it = cachedIsoScores.find({ops.first, ops.second, maxDepth});
    if (it != cachedIsoScores.end()) {
      return it->second;
    }
  }

  int64_t score = 0;
  if (visitedOps.find(ops) != visitedOps.end() || maxDepth == 0 ||
      ops.first->settings.recomputeType != ops.second->settings.recomputeType ||
      ops.first->scheduledPreLoss != ops.second->scheduledPreLoss ||
      (ops.first->getOptionalExecutionPhase() !=
       ops.second->getOptionalExecutionPhase()) ||
      (ops.first->getOptionalPipelineStage() !=
       ops.second->getOptionalPipelineStage())) {
    return score;
  }

  auto sit0 = opToSection.find(ops.first);
  auto sit1 = opToSection.find(ops.second);

  if (sit0 != opToSection.end() && sit1 != opToSection.end()) {
    if (sit0->second != sit1->second) {
      // Ops already annotated in different sections: Not isomorphic
      return score;
    } else {
      auto pos0 =
          opToPosition.at({sit0->second, ops.first->getBatchSerializedPhase()})
              .at(ops.first);
      auto pos1 =
          opToPosition.at({sit1->second, ops.second->getBatchSerializedPhase()})
              .at(ops.second);
      if (pos0 != pos1) {
        // Ops already annotated in same section but different positions: Not
        // isomorphic
      }
    }
  }

  visitedOps.insert(ops);

  // Check if the ops have the same subgraph equivalent ID
  if (opSubgraphEquivId[ops.first] == opSubgraphEquivId[ops.second]) {
    // Possibly isomorphic
    ++score;

    for (auto &input : ops.first->input->tensorMap()) {
      Tensor *tfirst  = ops.first->input->tensor(input.first);
      Tensor *tsecond = ops.second->input->tensor(input.first);
      if (tfirst->hasProducer() && tsecond->hasProducer()) {
        Op *pfirst  = tfirst->getProducer();
        Op *psecond = tsecond->getProducer();
        if (opSubgraphEquivId[pfirst] == opSubgraphEquivId[psecond]) {
          score += getLocalIsoScore(cachedIsoScores,
                                    opToSection,
                                    opToPosition,
                                    opSubgraphEquivId,
                                    {pfirst, psecond},
                                    visitedOps,
                                    maxDepth - 1,
                                    false);
        }
      }
    }

    for (auto &output : ops.first->output->tensorMap()) {
      if (!ops.first->output->hasIndex(output.first) ||
          !ops.second->output->hasIndex(output.first)) {
        continue;
      }
      Tensor *tfirst  = ops.first->output->tensor(output.first);
      Tensor *tsecond = ops.second->output->tensor(output.first);

      auto csfirst  = tfirst->consumers.getOps();
      auto cssecond = tsecond->consumers.getOps();

      std::set<Op *, POpCmp> cssecondMatched;

      for (Op *cfirst : csfirst) {
        Op *maxOp         = nullptr;
        int64_t max_score = 0;
        std::set<std::pair<Op *, Op *>> maxVisitedOps;
        for (Op *csecond : cssecond) {
          // Find csecond that matches cfirst best
          if (opSubgraphEquivId[cfirst] == opSubgraphEquivId[csecond] &&
              cssecondMatched.find(csecond) == cssecondMatched.end()) {
            std::set<std::pair<Op *, Op *>> &localVisitedOps = visitedOps;
            int64_t local_score = getLocalIsoScore(cachedIsoScores,
                                                   opToSection,
                                                   opToPosition,
                                                   opSubgraphEquivId,
                                                   {cfirst, csecond},
                                                   localVisitedOps,
                                                   maxDepth - 1,
                                                   false);
            if (local_score > max_score) {
              max_score     = local_score;
              maxVisitedOps = localVisitedOps;
              maxOp         = csecond;
            }
          }
        }
        if (maxOp) {
          cssecondMatched.insert(maxOp);
          visitedOps.insert(maxVisitedOps.begin(), maxVisitedOps.end());
          score += max_score;
        }
      }
    }
  }

  if (cached) {
    cachedIsoScores[{ops.first, ops.second, maxDepth}] = score;
  }
  return score;
}

void BatchSerialize::addParallelizationConstraints(
    Graph &graph,
    const PositionsToOp &positionToOp) const {
  // For maximum overlap we want the IoTileCopy (IO to compute) and the
  // preceding RemoteLoadOps for phase N+1 to happen during the compute
  // stage of batch N. To achieve this, let's schedule the IoTileCopy (IO
  // to compute) for N+1 to happen IoTileCopy (compute to IO) for N.

  auto &ir      = graph.getIr();
  auto settings = ir.getSessionOptions().batchSerializationSettings;

  for (auto &positions : positionToOp) {
    auto section = positions.first.first;
    auto phase   = positions.first.second;

    auto addConstraint = [&](Op *batchNPlus1Op, Op *batchNOp) {
      std::stringstream ss;
      ss << "[BatchSerialize] Added parallelization constraint "
         << batchNPlus1Op->str() << " -> " << batchNOp->str()
         << " (section {}, BSP {}-{}).";
      logging::transform::debug(ss.str(), section, phase - 1, phase);
      graph.topoCons->insert(batchNPlus1Op, batchNOp, true);
    };

    if (phase > 0) {

      // If we're overlapping, lift our batch N+1's loading IoTileCopy before
      // batch N's saving tile copy. This will force RemoteLoadOps for batch
      // N+1 to happen prior to the saving of the results of N's compute phase.

      Op *batchNPlus1Op =
          getLastIoTileCopyToCompute(positionToOp, section, phase);
      Op *batchNOp = getFirstIoTileCopyToIo(positionToOp, section, phase - 1);

      if (batchNPlus1Op && batchNOp) {
        addConstraint(batchNPlus1Op, batchNOp);
      }

      // If we're in OverlapOnCompute mode, try and lift batch N+1's
      // RemoteLoadOps to the front batch N's compute phase.

      if (settings.batchSchedule ==
          BatchSerializationBatchSchedule::OverlapOnCompute) {
        batchNPlus1Op = getLastRemoteLoad(positionToOp, section, phase);
        batchNOp      = getFirstComputeOp(positionToOp, section, phase - 1);

        if (batchNPlus1Op && batchNOp) {
          addConstraint(batchNPlus1Op, batchNOp);
        }
      }
    }
  }
}

Op *BatchSerialize::getLastRemoteLoad(const PositionsToOp &positionsToOp,
                                      const Section section,
                                      const BatchSerializedPhase phase) const {

  auto findIt = positionsToOp.find(std::make_pair(section, phase));

  if (findIt == positionsToOp.end()) {
    return nullptr;
  }

  for (auto it = findIt->second.rbegin(); it != findIt->second.rend(); ++it) {
    auto op = it->second;
    if (op->isConvertibleTo<RemoteLoadOp>()) {
      return op;
    }
  }

  return nullptr;
}

Op *BatchSerialize::getLastIoTileCopyToCompute(
    const PositionsToOp &positionsToOp,
    const Section section,
    const BatchSerializedPhase phase) const {

  auto findIt = positionsToOp.find(std::make_pair(section, phase));

  if (findIt == positionsToOp.end()) {
    return nullptr;
  }

  for (auto it = findIt->second.rbegin(); it != findIt->second.rend(); ++it) {
    auto op = it->second;
    if ((op->isConvertibleTo<IoTileCopyOp>()) &&
        (op->settings.tileSet == TileSet::Compute)) {
      return op;
    }
  }

  return nullptr;
}

Op *BatchSerialize::getFirstIoTileCopyToIo(
    const PositionsToOp &positionsToOp,
    const Section section,
    const BatchSerializedPhase phase) const {

  const auto findIt = positionsToOp.find(std::make_pair(section, phase));

  if (findIt == positionsToOp.end()) {
    return nullptr;
  }

  for (auto it = findIt->second.begin(); it != findIt->second.end(); ++it) {
    auto op = it->second;
    if ((op->isConvertibleTo<IoTileCopyOp>()) &&
        (op->settings.tileSet == TileSet::IO)) {
      return op;
    }
  }

  return nullptr;
}

Op *BatchSerialize::getFirstComputeOp(const PositionsToOp &positionsToOp,
                                      const Section section,
                                      const BatchSerializedPhase phase) const {

  const auto findIt = positionsToOp.find(std::make_pair(section, phase));

  if (findIt == positionsToOp.end()) {
    return nullptr;
  }

  for (auto it = findIt->second.begin(); it != findIt->second.end(); ++it) {
    auto op = it->second;
    if ((!op->isConvertibleTo<RemoteLoadOp>()) &&
        (!op->isConvertibleTo<IoTileCopyOp>()) &&
        (!op->isConvertibleTo<RemoteStoreOp>())) {
      return op;
    }
  }

  return nullptr;
}

bool BatchSerialize::areSwappable(Graph &graph,
                                  Op *earlierOp,
                                  Op *laterOp) const {

  if (graph.topoCons->contains(earlierOp, laterOp)) {
    // Don't go against topological constraints.
    return false;
  }

  if (earlierOp->isParentOf(laterOp)) {
    // Don't reverse direct parent/child relationships.
    return false;
  }

  return true;
}

void BatchSerialize::pushEarlier(
    PositionsToOpVector &vec,
    std::function<bool(Op *)> isPushOp,
    std::function<bool(Op *)> considerSwappingWith,
    std::function<bool(Op *, Op *)> areSwappable) const {
  // Move select to the front where this is possible.
  for (auto it1 = vec.begin(); it1 != vec.end(); ++it1) {

    if (isPushOp(it1->second)) {
      // Push selected op the furthest forward that we can.
      auto opIt = it1;

      while (true) {
        if (opIt == vec.begin()) {
          std::stringstream ss;
          ss << "[BatchSerialize] Considered moving " << opIt->second->str()
             << " earlier in the schedule "
             << "but no ops are available at earlier positions.";
          logging::transform::trace(ss.str());
          break;
        }

        // Get the op previous to ours.
        auto prevOpIt = opIt;
        prevOpIt--;

        // Check if we should consider swapping.
        if (!considerSwappingWith(prevOpIt->second)) {
          std::stringstream ss;
          ss << "[BatchSerialize] Not considering swapping "
             << opIt->second->str() << " with " << prevOpIt->second->str()
             << " (which is earlier in the schedule).";
          logging::transform::trace(ss.str());
          break;
        }

        // Check if we can swap.
        if (!areSwappable(prevOpIt->second, opIt->second)) {
          std::stringstream ss;
          ss << "[BatchSerialize] Not able to swap " << opIt->second->str()
             << " with " << prevOpIt->second->str()
             << " (which is earlier in the schedule) "
             << "as the ops are not swappable.";
          logging::transform::trace(ss.str());
          break;
        }

        std::stringstream ss;
        ss << "[BatchSerialize] Moving " << opIt->second->str()
           << " earlier in the schedule by "
           << "swapping it with " << prevOpIt->second->str()
           << " (in an attempt to make the "
           << "schedule more amenable to parallelization)";
        logging::transform::debug(ss.str());

        // OK, we can swap it!
        auto prevOp      = prevOpIt->second;
        prevOpIt->second = opIt->second;
        opIt->second     = prevOp;

        // We're now at the location of prevOpIt.
        opIt = prevOpIt;
      }
    }
  }
}

void BatchSerialize::pushLater(
    PositionsToOpVector &vec,
    std::function<bool(Op *)> isPushOp,
    std::function<bool(Op *)> considerSwappingWith,
    std::function<bool(Op *, Op *)> areSwappable) const {
  // Move select to the front where this is possible.
  for (auto it1 = vec.rbegin(); it1 != vec.rend(); ++it1) {

    if (isPushOp(it1->second)) {
      // Push selected op the furthest forward that we can.
      auto opIt = it1;

      while (true) {
        if (opIt == vec.rbegin()) {
          std::stringstream ss;
          ss << "[BatchSerialize] Considered moving " << opIt->second->str()
             << " later in the schedule "
             << "but no ops are available at later positions.";
          logging::transform::trace(ss.str());
          break;
        }

        // Get the op previous to ours.
        auto nextOpIt = opIt;
        nextOpIt--;

        // Check if we should consider swapping.
        if (!considerSwappingWith(nextOpIt->second)) {
          std::stringstream ss;
          ss << "[BatchSerialize] Not considering swapping "
             << opIt->second->str() << " with " << nextOpIt->second->str()
             << " (which is later in the schedule).";
          logging::transform::trace(ss.str());
          break;
        }

        // Check if we can swap.
        if (!areSwappable(opIt->second, nextOpIt->second)) {
          std::stringstream ss;
          ss << "[BatchSerialize] Not able to swap " << opIt->second->str()
             << " with " << nextOpIt->second->str()
             << " (which is later in the schedule) "
             << "as the ops are not swappable.";
          logging::transform::trace(ss.str());
          break;
        }

        std::stringstream ss;
        ss << "[BatchSerialize] Moving " << opIt->second->str()
           << " later in the schedule by "
           << "swapping it with " << nextOpIt->second->str()
           << " (in an attempt to make the "
           << "schedule more amenable to parallelization)";
        logging::transform::debug(ss.str());

        // OK, we can swap it!
        auto nextOp      = nextOpIt->second;
        nextOpIt->second = opIt->second;
        opIt->second     = nextOp;

        // We're now at the location of prevOpIt.
        opIt = nextOpIt;
      }
    }
  }
}

void BatchSerialize::tryToMakeAmenableToParallelization(
    Graph &graph,
    PositionsToOp &positionsToOp) const {

  // Predicate function to test if an op is a RemoteLoadOp.
  auto isRemoteLoad = [](Op *op) -> bool {
    return (op->isConvertibleTo<RemoteLoadOp>());
  };
  // Predicate function to test if an op is a RemoteLoadStore.
  auto isRemoteStore = [](Op *op) -> bool {
    return (op->isConvertibleTo<RemoteStoreOp>());
  };
  // Predicate function to test if an op is an IoTileCopyOp (from IO to
  // compute).
  auto isIoTileCopyToCompute = [](Op *op) -> bool {
    return (op->isConvertibleTo<IoTileCopyOp>()) &&
           (op->settings.tileSet == TileSet::Compute);
  };
  // Predicate function to test if an op is an IoTileCopyOp (from compute to
  // IO).
  auto isIoTileCopyToIo = [](Op *op) -> bool {
    return (op->isConvertibleTo<IoTileCopyOp>()) &&
           (op->settings.tileSet == TileSet::IO);
  };
  auto areSwappableL = [&](Op *earlierOp, Op *laterOp) {
    return areSwappable(graph, earlierOp, laterOp);
  };

  for (auto &entry : positionsToOp) {
    auto &posOpMap = entry.second;

    // Turn the map into a vector (it's easier to work with).
    PositionsToOpVector posOpVec;
    posOpVec.reserve(posOpMap.size());
    for (auto &posOpMapEntry : posOpMap) {
      posOpVec.push_back(
          std::make_pair(posOpMapEntry.first, posOpMapEntry.second));
    }

    // Move any RemoteLoadOp to the start of the schedule if possible, but don't
    // move past other RemoteLoadOps.
    pushEarlier(
        posOpVec,
        isRemoteLoad,
        [&](Op *op) { return !isRemoteLoad(op); },
        areSwappableL);

    // Move any IoTileCopyOp (from IO to compute) to the start, too. Don't move
    // past other IoTileCopyOp (from IO) ops or any RemoteLoadOps.
    pushEarlier(
        posOpVec,
        isIoTileCopyToCompute,
        [&](Op *op) { return !isRemoteLoad(op) && !isIoTileCopyToCompute(op); },
        areSwappableL);

    // Move any RemoteStoreOp back (later) to the back of the schedule if
    // possible, but don't move past other RemoteStoreOp.
    pushLater(
        posOpVec,
        isRemoteStore,
        [&](Op *op) { return !isRemoteStore(op); },
        areSwappableL);

    // Move any IoTileCopyOp (from IO to compute) to the back of the schedule if
    // possible. Don't move past other IoTileCopyOp (from IO) op or
    // RemoteStoreOps.
    pushLater(
        posOpVec,
        isIoTileCopyToIo,
        [&](Op *op) { return !isRemoteStore(op) && !isIoTileCopyToIo(op); },
        areSwappableL);

    // Turn the vector back into a map.
    posOpMap.clear();
    for (auto &posOpVecEntry : posOpVec) {
      posOpMap[posOpVecEntry.first] = posOpVecEntry.second;
    }
  }
}

namespace {
// BatchSerialize
// BatchSerialize 1: Copy ops to serialize forward pass, and add slices/concats
bool init1 = Transform::registerTransform(new BatchSerialize(1));
// BatchSerialize 2: Crystallize schedule
bool init2 = Transform::registerTransform(new BatchSerialize(2));
} // namespace

} // namespace popart
