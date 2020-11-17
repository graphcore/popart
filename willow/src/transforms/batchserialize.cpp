// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include <transforms/batchserialscheduler.hpp>
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
#include <popart/opsharding.hpp>
#include <popart/shardingplan.hpp>
#include <popart/tensor.hpp>
#include <popart/tensors.hpp>
#include <popart/topocons.hpp>
#include <popart/transforms/batchserialize.hpp>

namespace popart {

std::size_t BatchSerialize::id(int pass) {
  return typeid(BatchSerialize).hash_code() + pass;
}

bool BatchSerialize::apply(Graph &graph) const {
  logging::transform::debug("[BatchSerialize] Started.");

  ShardingHelper helper(&graph);

  std::set<Op *> toErase;

  auto &ir               = graph.getIr();
  auto settings          = ir.getSessionOptions().batchSerializationSettings;
  int64_t batchSerFactor = settings.factor;
  std::map<std::pair<TensorId, BatchSerialTensorContext>, std::set<OpId>>
      batchSerialOps;

  // FWD
  if (pass == 1) {
    auto schedule = graph.getOpSchedule({}, RequireOptimalSchedule::No);

    std::set<TensorId> tensorsWithBatch;
    std::set<Op *> serializedOps;
    std::map<std::pair<TensorId, BatchSerialTensorContext>,
             BatchSerializedTensorInfo>
        serializedTensorMap;

    for (TensorId id : ir.getTensorIds(TensorType::Stream)) {
      tensorsWithBatch.insert(id);
    }

    for (Op *op : schedule) {
      // Context in which the tensors are consumed
      BatchSerialTensorContext consumerContext =
          getBatchSerialTensorContext(op);
      Op::Settings consumerSettings = op->settings;

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
      if (!op->canShard()) {
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

        auto producerContext =
            entry.first->hasProducer()
                ? getBatchSerialTensorContext(entry.first->getProducer())
                : BatchSerialTensorContext();

        auto serializedItProducer =
            serializedTensorMap.find({entry.first->id, producerContext});
        auto serializedItConsumer =
            serializedTensorMap.find({entry.first->id, consumerContext});

        bool isRemoteArg = entry.first->isRemoteArgTensor();

        bool hasBatch =
            tensorsWithBatch.find(entry.first->id) != tensorsWithBatch.end() ||
            (isProducerInitOp && entry.first->getBatchAxis() != -1) ||
            isRemoteArg;

        logging::transform::trace(
            "[BatchSerialize] input tensor {} type: {} shape: {} "
            "serialized: [p: {} c: {} h: {}]",
            entry.first->id,
            type,
            shape,
            serializedItProducer != serializedTensorMap.end(),
            serializedItConsumer != serializedTensorMap.end(),
            hasBatch);

        // a.) Tensor has special handling rules (e.g. remoteArg)
        // b.) Tensor can be serialized on the batch dimension
        // c.) Tensor has no producer, or is not yet registered in the
        // serialized tensor map
        if (isRemoteArg &&
            (serializedItProducer == serializedTensorMap.end() ||
             serializedItConsumer == serializedTensorMap.end())) {

          TensorId remoteArgId = entry.first->id;

          BatchSerializedTensorInfo &bsInfo =
              serializedTensorMap[{entry.first->id, producerContext}];
          bsInfo.id   = remoteArgId;
          bsInfo.info = entry.first->info;

          for (int64_t b = 0; b < batchSerFactor; ++b) {
            auto remoteArgBsId = ir.createSliceTensorId(remoteArgId, b, b + 1);
            TensorInfo argTensorInfo(DataType::INT32, {1});
            std::vector<int> idData(1, 0);
            graph.getTensors().addConstInit(
                remoteArgBsId,
                argTensorInfo,
                reinterpret_cast<void *>(idData.data()));

            bsInfo.serializedIds.push_back(remoteArgBsId);
            bsInfo.serializedInfos.push_back(argTensorInfo);
          }

          hasBatch = true;
        } else if (hasBatch &&
                   (!entry.first->hasProducer() ||
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
                          "batch serialization factor: {} for tensor {}",
                          shape[axis],
                          batchSerFactor,
                          entry.first->id);
            }
            logging::transform::trace(
                "[BatchSerialize] Batch axis for {} is {}",
                entry.first->id,
                axis);

            if (consumerContext.vgraphId) {
              consumerSettings.vgraphId = consumerContext.vgraphId;
            }
            if (consumerContext.executionPhase) {
              consumerSettings.executionPhase = consumerContext.executionPhase;
            }
            if (consumerContext.pipelineStage) {
              consumerSettings.pipelineStage = consumerContext.pipelineStage;
            }

            BatchSerializedTensorInfo &bsInfo =
                serializedTensorMap[{entry.first->id, consumerContext}];
            bsInfo.id   = entry.first->id;
            bsInfo.info = entry.first->info;

            std::vector<Op::Settings> sliceSettings(batchSerFactor,
                                                    consumerSettings);
            for (int64_t b = 0; b < batchSerFactor; ++b) {
              TensorId sliceId =
                  ir.createSliceTensorId(entry.first->id,
                                         static_cast<unsigned int>(b),
                                         static_cast<unsigned int>(b + 1));
              bsInfo.serializedIds.push_back(sliceId);
              sliceSettings.at(b).batchSerializedPhase = b;
            }

            std::vector<Op *> sops;
            if (settings.method == BatchSerializationMethod::UnrollDynamic) {
              auto preSettings                 = consumerSettings;
              preSettings.batchSerializedPhase = OptionalBatchSerializedPhase();
              auto postSettings                = consumerSettings;
              postSettings.batchSerializedPhase =
                  OptionalBatchSerializedPhase();
              sliceSettings.push_back(preSettings);
              sliceSettings.push_back(postSettings);
              sops = helper.dynamicShard(
                  axis, bsInfo.serializedIds, entry.first->id, sliceSettings);
              for (int64_t b = 0; b < batchSerFactor; ++b) {
                bsInfo.serializedInfos.push_back(
                    graph.getTensors().get(bsInfo.serializedIds.at(b))->info);
              }
            } else if (settings.method ==
                       BatchSerializationMethod::UnrollStatic) {
              sops = helper.staticShard(
                  axis, bsInfo.serializedIds, entry.first->id, sliceSettings);
              for (int64_t b = 0; b < batchSerFactor; ++b) {
                bsInfo.serializedInfos.push_back(
                    graph.getTensors().get(bsInfo.serializedIds.at(b))->info);
              }
            } else if (settings.method == BatchSerializationMethod::Loop) {
              bsInfo.serializedIds.clear();
              TensorInfo info = bsInfo.info;
              Shape shape     = info.shape();
              shape.at(axis) /= batchSerFactor;
              info.set(info.dataType(), shape);
              for (int64_t b = 0; b < batchSerFactor; ++b) {
                bsInfo.serializedInfos.push_back(info);
              }
            } else {
              throw error("[BatchSerialize] Unsupported method.");
            }
            for (Op *sop : sops) {
              batchSerialOps[{entry.first->id, consumerContext}].insert(
                  sop->id);
            }
          }
        } else if ((serializedItProducer != serializedTensorMap.end() &&
                    serializedItProducer->second.serializedInfos.size() ==
                        batchSerFactor) ||
                   (serializedItConsumer != serializedTensorMap.end() &&
                    serializedItConsumer->second.serializedInfos.size() ==
                        batchSerFactor)) {
          // Input already serialized
          op_has_batch |= true;
        }
      }

      // Operations not affected by the batch size can skip this part
      if (op_has_batch) {
        ShardIdMap shardInputs;
        ShardInfoMap shardInfos;

        for (auto &in : op->input->tensorMap()) {
          auto serializedTensor =
              serializedTensorMap.find({in.second->id, consumerContext});
          if (serializedTensor != serializedTensorMap.end()) {
            // Tensors split along batch dimension
            if (serializedTensor->second.serializedIds.size() ==
                batchSerFactor) {
              for (int64_t b = 0; b < batchSerFactor; ++b) {
                shardInputs[in.second->id].push_back(
                    serializedTensor->second.serializedIds.at(b));
              }
            } else {
              // Tensor not split along batch dimension, but split info
              // available
              std::vector<TensorInfo> infos;
              for (int64_t b = 0; b < batchSerFactor; ++b) {
                infos.push_back(serializedTensor->second.serializedInfos.at(b));
              }
              shardInfos[in.second->id] = {
                  serializedTensor->second.concatId.empty()
                      ? serializedTensor->second.id
                      : serializedTensor->second.concatId,
                  serializedTensor->second.info,
                  infos};
            }
          }
        }

        ShardingMethod method;
        switch (settings.method) {
        case BatchSerializationMethod::UnrollDynamic: {
          method = ShardingMethod::DynamicShard;
          break;
        };
        case BatchSerializationMethod::UnrollStatic: {
          method = ShardingMethod::StaticShard;
          break;
        };
        case BatchSerializationMethod::Loop: {
          method = ShardingMethod::Loop;
          break;
        };
        }

        ShardingPlan outputPlan(method);
        // The following will throw an error if batch serialisation failed
        // to slice a tensor. Return a sensible error message.
        try {
          std::vector<Op::Settings> shardSettings(batchSerFactor, op->settings);
          for (int64_t b = 0; b < batchSerFactor; ++b) {
            shardSettings.at(b).batchSerializedPhase = b;
          }

          ShardOpSettings shardOpSettings;
          shardOpSettings.setPreSettings(op->settings);
          shardOpSettings.setShardSettings(shardSettings);
          shardOpSettings.setPostSettings(op->settings);

          ShardingPlan inputPlan(method, shardOpSettings);
          inputPlan.insertIdMap(shardInputs, graph);
          inputPlan.insertInfoMap(shardInfos);

          outputPlan = op->shard(inputPlan);

        } catch (error &e) {
          std::stringstream ss;
          ss << "Batch serialisation failed while processing op " << op->opid;
          ss << ". The inputs to this op are: ";
          for (unsigned j = 0; j < op->inTensorCount(); j++) {
            ss << op->inId(j) << ((j < op->inTensorCount() - 1) ? ", " : ".");
          }
          throw error(ss.str());
        }

        // Output tensor shardable, but no sharded tensor IDs
        for (auto &idkv : outputPlan.getInfoMap()) {
          serializedTensorMap[{idkv.first, consumerContext}].id = idkv.first;
          serializedTensorMap[{idkv.first, consumerContext}].concatId =
              std::get<0>(idkv.second);
          serializedTensorMap[{idkv.first, consumerContext}].info =
              std::get<1>(idkv.second);

          if (std::get<2>(idkv.second).size() == batchSerFactor) {
            serializedTensorMap[{idkv.first, consumerContext}].serializedInfos =
                std::get<2>(idkv.second);
            logging::trace("[BatchSerialize] Tensor: {} {} shards (info).",
                           idkv.first,
                           batchSerFactor);
          } else {
            tensorsWithBatch.erase(idkv.first);
            logging::trace("[BatchSerialize] Tensor: {} no shards (info).",
                           idkv.first);
          }
        }

        // Output tensor shardable, with sharded tensor IDs
        for (auto &idkv : outputPlan.getIdMap()) {
          if (idkv.second.size() == batchSerFactor) {
            serializedTensorMap[{idkv.first, consumerContext}].serializedIds =
                idkv.second;
            serializedTensorMap[{idkv.first, consumerContext}].concatId.clear();
            logging::trace("[BatchSerialize] Tensor: {} {} shards (ids).",
                           idkv.first,
                           batchSerFactor);
          } else {
            logging::trace("[BatchSerialize] Tensor: {} no shards (ids).",
                           idkv.first);
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
      auto producerContext = getBatchSerialTensorContext(producer);
      if (serializedTensor.first.second != producerContext) {
        continue;
      }

      auto concatIfNecessary = [&settings,
                                &tensor,
                                &producer,
                                &ir,
                                &producerContext,
                                &serializedTensor,
                                &serializedTensorMap,
                                &batchSerialOps,
                                &helper]() {
        if (serializedTensor.second.concatId.empty()) {
          // TODO T20169: Different axis support
          TensorId serId0 = serializedTensor.second.serializedIds.at(0);
          Tensor *serT0   = ir.getTensor(serId0);
          int64_t axis    = 0;
          for (unsigned i = 0; i < tensor->info.shape().size(); ++i) {
            if (serT0->info.shape()[i] < tensor->info.shape()[i]) {
              axis = i;
              break;
            }
          }

          auto &tensorIds = serializedTensor.second.serializedIds;

          TensorId concatId = serializedTensor.first.first;

          auto producerSettings = producer->getOutSettings(
              producer->output->indices(tensor).front());

          if (producerContext.vgraphId) {
            producerSettings.vgraphId = producerContext.vgraphId;
          }
          if (producerContext.executionPhase) {
            producerSettings.executionPhase = producerContext.executionPhase;
          }
          if (producerContext.pipelineStage) {
            producerSettings.pipelineStage = producerContext.pipelineStage;
          }

          if (settings.method == BatchSerializationMethod::UnrollDynamic ||
              settings.method == BatchSerializationMethod::Loop) {
            std::vector<Op::Settings> concatSettings(tensorIds.size() + 2,
                                                     producerSettings);

            for (size_t b = 0; b < tensorIds.size(); ++b) {
              concatSettings.at(b).batchSerializedPhase = b;
            }
            concatSettings.at(tensorIds.size()).batchSerializedPhase =
                OptionalBatchSerializedPhase();
            concatSettings.at(tensorIds.size() + 1).batchSerializedPhase =
                OptionalBatchSerializedPhase();

            std::vector<Op *> sops =
                helper.dynamicConcat(axis, tensorIds, concatId, concatSettings);
            for (Op *sop : sops) {
              batchSerialOps[{tensor->id, producerContext}].insert(sop->id);
            }
          } else {
            Op *concatOp =
                helper.staticConcat(axis, tensorIds, concatId, producerSettings)
                    .front();
            batchSerialOps[{tensor->id, producerContext}].insert(concatOp->id);
          }

          serializedTensorMap[{serializedTensor.first.first, producerContext}]
              .concatId = concatId;
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
          if (IpuCopyOp *copyOp = dynamic_cast<IpuCopyOp *>(consumer)) {
            auto source =
                copyOp->getSourceIpu(copyOp->input->tensorIdMap().at(i));
            consumer->disconnectInTensor(i, tensor);
            copyOp->connectInTensor(
                i,
                serializedTensorMap.at({tensor->id, producerContext}).concatId,
                source);
          } else {
            consumer->disconnectInTensor(i, tensor);
            consumer->connectInTensor(
                i,
                serializedTensorMap.at({tensor->id, producerContext}).concatId);
          }
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

    // Reset fromLoss/toLoss on everything except the final loss
    for (auto &opIdAndOp : graph.getOps()) {
      if (!(opIdAndOp.second->toLoss == PathToLoss::Yes &&
            opIdAndOp.second->fromLoss == PathFromLoss::Yes)) {
        opIdAndOp.second->toLoss   = PathToLoss::Undefined;
        opIdAndOp.second->fromLoss = PathFromLoss::Undefined;
        for (auto &t : opIdAndOp.second->input->tensorMap()) {
          t.second->toLoss   = PathToLoss::Undefined;
          t.second->fromLoss = PathFromLoss::Undefined;
        }
        for (auto &t : opIdAndOp.second->output->tensorMap()) {
          t.second->toLoss   = PathToLoss::Undefined;
          t.second->fromLoss = PathFromLoss::Undefined;
        }
      }
    }
  }

  // Annotate priorities to isolate batch ops and crystallize the schedule
  // between batch serial phases
  if (pass == 2) {

    // If batchSchedule == Scheduler we defer any further scheduling to the
    // scheduler.
    if (settings.batchSchedule != BatchSerializationBatchSchedule::Scheduler &&
        settings.method != BatchSerializationMethod::Loop) {
      BatchSerialScheduler scheduler(graph);
      scheduler.apply();
    }

    // Freeze the schedule completely, so that the batch serial order cannot be
    // undone by outlining
    graph.freezeSchedule({});
  }

  logging::transform::debug("[BatchSerialize] Done.");
  return true;
}

namespace {
// BatchSerialize
// BatchSerialize 1: Copy ops to serialize forward/backward pass, and add
// slices/concats
bool init1 = Transform::registerTransform(new BatchSerialize(1));
// BatchSerialize 2: Crystallize schedule
bool init2 = Transform::registerTransform(new BatchSerialize(2));
} // namespace

} // namespace popart
