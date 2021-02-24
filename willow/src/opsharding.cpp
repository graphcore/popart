// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <onnx/onnx_pb.h>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op.hpp>
#include <popart/op/add.hpp>
#include <popart/op/cast.hpp>
#include <popart/op/concat.hpp>
#include <popart/op/div.hpp>
#include <popart/op/dynamic/dynamicslice.hpp>
#include <popart/op/dynamic/dynamicupdate.hpp>
#include <popart/op/elementwise.hpp>
#include <popart/op/identity.hpp>
#include <popart/op/init.hpp>
#include <popart/op/ipucopy.hpp>
#include <popart/op/loop.hpp>
#include <popart/op/mean.hpp>
#include <popart/op/reshape.hpp>
#include <popart/op/restore.hpp>
#include <popart/op/scale.hpp>
#include <popart/op/slice.hpp>
#include <popart/op/sum.hpp>
#include <popart/opsharding.hpp>
#include <popart/shardingplan.hpp>
#include <popart/tensor.hpp>
#include <popart/tensordata.hpp>
#include <popart/tensorindex.hpp>
#include <popart/tensors.hpp>
#include <popart/topocons.hpp>
#include <popart/util.hpp>

namespace popart {

ShardingHelper::ShardingHelper(Graph *graph_) : graph(graph_) {}

std::vector<Op *> ShardingHelper::staticConcat(int64_t axis,
                                               std::vector<TensorId> tensorIds,
                                               TensorId concatId,
                                               Op::Settings settings) const {
  std::unique_ptr<ConcatOp> concatOpUp =
      std::make_unique<ConcatOp>(Onnx::AiOnnx::OpSet11::Concat, axis, settings);
  ConcatOp *concatOp = concatOpUp.get();
  concatOp->setName("Concat_" + concatId);
  graph->moveIntoGraph(std::move(concatOpUp));

  for (size_t b = 0; b < tensorIds.size(); ++b) {
    concatOp->connectInTensor(static_cast<popart::InIndex>(b), tensorIds.at(b));
  }
  concatOp->createAndConnectOutTensor(ConcatOp::getOutIndex(), concatId);
  concatOp->setup();
  return {concatOp};
}

void ShardingHelper::connectOutTensor(Op *op,
                                      TensorId id,
                                      OutIndex index) const {
  auto &ir = op->getIr();
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

std::vector<Op *> ShardingHelper::reshapeForSlice(TensorId inId,
                                                  Shape newShape,
                                                  TensorId outId,
                                                  Op::Settings settings) const {
  logging::transform::trace(
      "[ShardingHelper] Reshaping {} -> {} {}", inId, outId, newShape);
  std::unique_ptr<ReshapeOp> reshapeOpUp = std::make_unique<ReshapeOp>(
      Onnx::AiOnnx::OpSet11::Reshape, newShape, settings);
  Op *reshapeOp = reshapeOpUp.get();
  graph->moveIntoGraph(std::move(reshapeOpUp));

  reshapeOp->setName("Reshape_" + inId);
  reshapeOp->connectInTensor(ReshapeOp::getInIndex(), inId);
  connectOutTensor(reshapeOp, outId, ReshapeOp::getOutIndex());
  reshapeOp->setup();
  return {reshapeOp};
}

std::vector<Op *>
ShardingHelper::staticShard(int64_t axis,
                            std::vector<TensorId> tensorIds,
                            TensorId concatId,
                            std::vector<Op::Settings> settings) const {
  if (settings.size() != tensorIds.size() && settings.size() != 1) {
    throw error("[ShardingHelper] Expected {} or 1 settings.",
                tensorIds.size());
  }

  std::vector<Op *> ops;

  auto t = graph->getIr().getTensor(concatId);

  int64_t slice_size = t->info.shape().at(axis) / tensorIds.size();

  for (int64_t b = 0; b < tensorIds.size(); ++b) {
    std::vector<int64_t> startsv(1, b * slice_size);
    std::vector<int64_t> endsv(1, (b + 1) * slice_size);
    std::vector<int64_t> axesv(1, axis);

    std::unique_ptr<SliceOp> sliceOpUp =
        std::make_unique<SliceOp>(Onnx::AiOnnx::OpSet11::Slice,
                                  startsv,
                                  endsv,
                                  axesv,
                                  std::vector<int64_t>{}, // steps
                                  settings.at(b % settings.size()));
    SliceOp *sliceOp = sliceOpUp.get();
    graph->moveIntoGraph(std::move(sliceOpUp));
    sliceOp->setName("Slice_" + tensorIds.at(b));
    sliceOp->createAndConnectOutTensor(SliceOp::getOutIndex(), tensorIds.at(b));
  }
  return ops;
}

std::vector<Op *>
ShardingHelper::dynamicConcat(int64_t axis,
                              std::vector<TensorId> tensorIds,
                              TensorId concatId,
                              std::vector<Op::Settings> settings) const {

  if (settings.size() != tensorIds.size() + 2 && settings.size() != 1) {
    throw error("[ShardingHelper] Expected {} or 1 settings.",
                tensorIds.size() + 2);
  }

  TensorId lastId;

  std::vector<Op *> ops;

  for (size_t b = 0; b < tensorIds.size(); ++b) {
    TensorId sliceTensorId = tensorIds.at(b);
    Tensor *s              = graph->getTensors().get(sliceTensorId);

    auto origSliceShape = s->info.shape();
    if (origSliceShape.empty()) {
      origSliceShape.push_back(1);
    }

    auto outShape   = origSliceShape;
    auto initShape  = origSliceShape;
    auto sliceShape = origSliceShape;

    TensorId toUpdateSliceTensorId;
    if (origSliceShape[axis] > 1) {
      initShape.resize(initShape.size() + 1);
      sliceShape.resize(sliceShape.size() + 1);
      for (size_t i = 0; i < initShape.size(); ++i) {
        if (i < axis) {
          // Shape remains
        } else if (i == axis) {
          initShape[i]  = tensorIds.size();
          sliceShape[i] = 1;
          outShape[i]   = tensorIds.size() * origSliceShape[i];
        } else if (i == axis + 1) {
          initShape[i]  = origSliceShape[i - 1];
          sliceShape[i] = origSliceShape[i - 1];
        } else if (i > axis + 1) {
          initShape[i]  = origSliceShape[i - 1];
          sliceShape[i] = origSliceShape[i - 1];
        }
      }

      logging::transform::trace(
          "[ShardingHelper] Reshape for update: [{} -> {}, {}]",
          outShape,
          initShape,
          sliceShape);

      toUpdateSliceTensorId =
          graph->getIr().createIntermediateTensorId(sliceTensorId);
      auto reshapeOps = reshapeForSlice(sliceTensorId,
                                        sliceShape,
                                        toUpdateSliceTensorId,
                                        settings.at(b % settings.size()));
      ops.insert(ops.end(), reshapeOps.begin(), reshapeOps.end());
    } else {
      toUpdateSliceTensorId = sliceTensorId;
      initShape[axis]       = tensorIds.size() * origSliceShape[axis];
      outShape[axis]        = tensorIds.size() * origSliceShape[axis];
    }

    if (b == 0) {
      TensorInfo info = graph->getIr().getTensor(tensorIds.at(0))->info;
      info.set(info.dataType(), initShape);
      Tensor *init =
          initTensor(info,
                     concatId,
                     InitType::NoInit,
                     settings.at((settings.size() - 2) % settings.size()));
      Op *initOp = init->getProducer();
      lastId     = init->id;
      ops.push_back(initOp);
    }

    std::vector<int64_t> axesv(1, axis);
    std::vector<int64_t> sizesv(1, 1);

    std::unique_ptr<DynamicUpdateOp> updateOpUp =
        std::make_unique<DynamicUpdateOp>(
            Onnx::CustomOperators::DynamicUpdate_1,
            axesv,
            sizesv,
            true,
            settings.at(b % settings.size()));
    DynamicUpdateOp *updateOp = updateOpUp.get();
    graph->moveIntoGraph(std::move(updateOpUp));
    updateOp->setName("Concat_" + concatId);

    updateOp->connectInTensor(DynamicUpdateOp::getInIndex(),
                              toUpdateSliceTensorId);
    updateOp->connectInTensor(
        DynamicUpdateOp::getIndexInIndex(),
        createOrGetIndexTensor(static_cast<uint32_t>(b),
                               settings.at(b % settings.size())));
    updateOp->connectInTensor(DynamicUpdateOp::getUpdateInIndex(), lastId);

    updateOp->settings.inferTensorMappingToFrom.insert(
        {DynamicUpdateOp::getUpdateInIndex(), DynamicUpdateOp::getInIndex()});

    lastId = (b == tensorIds.size() - 1 && origSliceShape[axis] == 1)
                 ? concatId
                 : graph->getIr().createIntermediateTensorId(concatId);
    connectOutTensor(updateOp, lastId, DynamicUpdateOp::getOutIndex());
    updateOp->setup();

    if (b == tensorIds.size() - 1 && origSliceShape[axis] > 1) {

      logging::transform::trace(
          "[ShardingHelper] Reshape after last update: [{} -> {}]",
          initShape,
          outShape);

      auto reshapeOps =
          reshapeForSlice(lastId,
                          outShape,
                          concatId,
                          settings.at((settings.size() - 1) % settings.size()));
      ops.insert(ops.end(), reshapeOps.begin(), reshapeOps.end());
    }
  }
  return ops;
}

std::vector<Op *>
ShardingHelper::dynamicShard(int64_t axis,
                             std::vector<TensorId> tensorIds,
                             TensorId concatId,
                             std::vector<Op::Settings> settings) const {
  if (settings.size() != tensorIds.size() + 2 && settings.size() != 1) {
    throw error("[ShardingHelper] Expected {} or 1 settings.",
                tensorIds.size() + 2);
  }

  Tensor *inTensor   = graph->getIr().getTensor(concatId);
  Shape origShape    = inTensor->info.shape();
  Shape sliceShape   = inTensor->info.shape();
  Shape reShape      = inTensor->info.shape();
  Shape sliceReShape = inTensor->info.shape();

  TensorId sliceableTensorId;
  std::vector<Op *> ops;

  // Reshape to minimize sliceable offsets along the axis dimension
  if (origShape[axis] > tensorIds.size()) {
    reShape.resize(reShape.size() + 1);
    sliceReShape.resize(sliceReShape.size() + 1);
    for (size_t i = 0; i < reShape.size(); ++i) {
      if (i < axis) {
        // Shapes remain
      } else if (i == axis) {
        reShape[i]      = tensorIds.size();
        sliceReShape[i] = 1;
        sliceShape[i]   = origShape[i] / tensorIds.size();
      } else if (i == axis + 1) {
        reShape[i]      = origShape[i - 1] / tensorIds.size();
        sliceReShape[i] = origShape[i - 1] / tensorIds.size();
      } else if (i > axis + 1) {
        reShape[i]      = origShape[i - 1];
        sliceReShape[i] = origShape[i - 1];
      }
    }

    logging::transform::trace(
        "[ShardingHelper] Reshape to sliceable: [{} -> {}]",
        origShape,
        reShape);

    sliceableTensorId = graph->getIr().createIntermediateTensorId(concatId);
    auto reshapeOps =
        reshapeForSlice(concatId,
                        reShape,
                        sliceableTensorId,
                        settings.at((settings.size() - 2) % settings.size()));
    ops.insert(ops.end(), reshapeOps.begin(), reshapeOps.end());

  } else {
    sliceableTensorId = concatId;
  }

  for (int64_t b = 0; b < tensorIds.size(); ++b) {

    std::vector<int64_t> axesv(1, axis);
    std::vector<int64_t> sizesv(1, 1);

    std::unique_ptr<DynamicSliceOp> sliceOpUp =
        std::make_unique<DynamicSliceOp>(Onnx::CustomOperators::DynamicSlice_1,
                                         axesv,
                                         sizesv,
                                         true,
                                         settings.at(b % settings.size()));
    DynamicSliceOp *sliceOp = sliceOpUp.get();
    graph->moveIntoGraph(std::move(sliceOpUp));
    sliceOp->setName("Slice_" + concatId);
    ops.push_back(sliceOp);

    sliceOp->connectInTensor(SliceOp::getInIndex(), sliceableTensorId);
    sliceOp->connectInTensor(
        DynamicSliceOp::getIndexInIndex(),
        createOrGetIndexTensor(static_cast<uint32_t>(b),
                               settings.at(b % settings.size())));

    TensorId tmpSliceId =
        origShape[axis] > tensorIds.size()
            ? graph->getIr().createIntermediateTensorId(tensorIds.at(b))
            : tensorIds.at(b);

    sliceOp->createAndConnectOutTensor(SliceOp::getOutIndex(), tmpSliceId);
    sliceOp->setup();

    logging::transform::trace("[ShardingHelper] Slice tensor {} {} -> {} {}",
                              concatId,
                              origShape,
                              tensorIds.at(b),
                              sliceReShape);

    if (origShape[axis] > tensorIds.size()) {

      logging::transform::trace("[ShardingHelper] Reshape slice: [{} -> {}]",
                                sliceReShape,
                                sliceShape);

      auto reshapeOps = reshapeForSlice(tmpSliceId,
                                        sliceShape,
                                        tensorIds.at(b),
                                        settings.at(b % settings.size()));
      ops.insert(ops.end(), reshapeOps.begin(), reshapeOps.end());
    }
  }

  return ops;
}

std::vector<Op *> ShardingHelper::dynamicSlice(int64_t axis,
                                               int64_t num_shards,
                                               TensorId sliceId,
                                               TensorId concatId,
                                               TensorId indexId,
                                               Op::Settings settings) const {
  Tensor *inTensor   = graph->getIr().getTensor(concatId);
  Shape origShape    = inTensor->info.shape();
  Shape sliceShape   = inTensor->info.shape();
  Shape reShape      = inTensor->info.shape();
  Shape sliceReShape = inTensor->info.shape();

  TensorId sliceableTensorId;
  std::vector<Op *> ops;

  // Reshape to minimize sliceable offsets along the axis dimension
  if (origShape[axis] > num_shards) {
    reShape.resize(reShape.size() + 1);
    sliceReShape.resize(sliceReShape.size() + 1);
    for (size_t i = 0; i < reShape.size(); ++i) {
      if (i < axis) {
        // Shapes remain
      } else if (i == axis) {
        reShape[i]      = num_shards;
        sliceReShape[i] = 1;
        sliceShape[i]   = origShape[i] / num_shards;
      } else if (i == axis + 1) {
        reShape[i]      = origShape[i - 1] / num_shards;
        sliceReShape[i] = origShape[i - 1] / num_shards;
      } else if (i > axis + 1) {
        reShape[i]      = origShape[i - 1];
        sliceReShape[i] = origShape[i - 1];
      }
    }

    logging::transform::trace(
        "[ShardingHelper] Reshape to sliceable: [{} -> {}]",
        origShape,
        reShape);

    sliceableTensorId = graph->getIr().createIntermediateTensorId(concatId);
    auto reshapeOps =
        reshapeForSlice(concatId, reShape, sliceableTensorId, settings);
    ops.insert(ops.end(), reshapeOps.begin(), reshapeOps.end());

  } else {
    sliceableTensorId = concatId;
  }

  std::vector<int64_t> axesv(1, axis);
  std::vector<int64_t> sizesv(1, 1);

  std::unique_ptr<DynamicSliceOp> sliceOpUp = std::make_unique<DynamicSliceOp>(
      Onnx::CustomOperators::DynamicSlice_1, axesv, sizesv, true, settings);
  DynamicSliceOp *sliceOp = sliceOpUp.get();
  graph->moveIntoGraph(std::move(sliceOpUp));
  sliceOp->setName("Slice_" + concatId);
  ops.push_back(sliceOp);

  sliceOp->connectInTensor(SliceOp::getInIndex(), sliceableTensorId);
  sliceOp->connectInTensor(DynamicSliceOp::getIndexInIndex(), indexId);

  TensorId tmpSliceId = origShape[axis] > num_shards
                            ? graph->getIr().createIntermediateTensorId(sliceId)
                            : sliceId;

  sliceOp->createAndConnectOutTensor(SliceOp::getOutIndex(), tmpSliceId);
  sliceOp->setup();

  logging::transform::trace("[ShardingHelper] Slice tensor {} {} -> {} {}",
                            concatId,
                            origShape,
                            tmpSliceId,
                            sliceShape);

  if (origShape[axis] > num_shards) {

    logging::transform::trace(
        "[ShardingHelper] Reshape slice: [{} -> {}]", sliceReShape, sliceShape);

    auto reshapeOps =
        reshapeForSlice(tmpSliceId, sliceShape, sliceId, settings);
    ops.insert(ops.end(), reshapeOps.begin(), reshapeOps.end());
  }

  return ops;
}

std::vector<Op *> ShardingHelper::dynamicUpdate(int64_t axis,
                                                int64_t num_shards,
                                                TensorId sliceId,
                                                TensorId concatInId,
                                                TensorId concatOutId,
                                                TensorId indexId,
                                                Op::Settings settings) const {
  std::vector<Op *> ops;

  Tensor *slice    = graph->getTensors().get(sliceId);
  Tensor *concatIn = graph->getTensors().get(concatInId);

  auto sliceShape   = slice->info.shape();
  auto origShape    = concatIn->info.shape();
  auto sliceReShape = slice->info.shape();
  auto inReShape    = concatIn->info.shape();

  TensorId sliceReId = sliceId;
  TensorId inReId    = concatInId;
  TensorId outReId   = concatOutId;

  if (sliceShape[axis] > 1) {
    sliceReShape.resize(origShape.size() + 1);
    inReShape.resize(sliceShape.size() + 1);
    for (size_t i = 0; i < sliceReShape.size(); ++i) {
      if (i < axis) {
        // Shape remains
      } else if (i == axis) {
        sliceReShape[i] = 1;
        inReShape[i]    = num_shards;
      } else if (i == axis + 1) {
        sliceReShape[i] = sliceShape[i - 1];
        inReShape[i]    = origShape[i - 1] / num_shards;
      } else if (i > axis + 1) {
        sliceReShape[i] = sliceShape[i - 1];
        inReShape[i]    = origShape[i - 1];
      }
    }

    logging::transform::trace(
        "[ShardingHelper] Reshape for update: [{} -> {}, {} -> {}]",
        sliceShape,
        sliceReShape,
        origShape,
        inReShape);

    sliceReId = graph->getIr().createIntermediateTensorId(sliceReId);
    inReId    = graph->getIr().createIntermediateTensorId(inReId);
    outReId   = graph->getIr().createIntermediateTensorId(outReId);
    {
      auto reshapeOps =
          reshapeForSlice(sliceId, sliceReShape, sliceReId, settings);
      ops.insert(ops.end(), reshapeOps.begin(), reshapeOps.end());
    }
    {
      auto reshapeOps =
          reshapeForSlice(concatInId, inReShape, inReId, settings);
      ops.insert(ops.end(), reshapeOps.begin(), reshapeOps.end());
    }
  }

  std::vector<int64_t> axesv(1, axis);
  std::vector<int64_t> sizesv(1, 1);

  std::unique_ptr<DynamicUpdateOp> updateOpUp =
      std::make_unique<DynamicUpdateOp>(Onnx::CustomOperators::DynamicUpdate_1,
                                        axesv,
                                        sizesv,
                                        true,
                                        settings);
  DynamicUpdateOp *updateOp = updateOpUp.get();
  graph->moveIntoGraph(std::move(updateOpUp));
  updateOp->setName("Update_" + concatInId);

  updateOp->connectInTensor(DynamicUpdateOp::getInIndex(), sliceReId);
  updateOp->connectInTensor(DynamicUpdateOp::getIndexInIndex(), indexId);
  updateOp->connectInTensor(DynamicUpdateOp::getUpdateInIndex(), inReId);

  updateOp->settings.inferTensorMappingToFrom.insert(
      {DynamicUpdateOp::getUpdateInIndex(), DynamicUpdateOp::getInIndex()});

  updateOp->createAndConnectOutTensor(DynamicUpdateOp::getOutIndex(), outReId);
  updateOp->setup();

  if (sliceShape[axis] > 1) {

    logging::transform::trace(
        "[ShardingHelper] Reshape after update: [{} -> {}]",
        inReShape,
        origShape);

    auto reshapeOps =
        reshapeForSlice(outReId, origShape, concatOutId, settings);
    ops.insert(ops.end(), reshapeOps.begin(), reshapeOps.end());
  }
  return ops;
}

std::vector<Op *> ShardingHelper::offset(TensorId tensorId,
                                         TensorId concatId,
                                         TensorId offsetId,
                                         Op::Settings settings) const {
  std::vector<Op *> ops;

  auto t  = graph->getIr().getTensor(concatId);
  auto to = graph->getIr().getTensor(offsetId);

  if (t->info.dataType() != to->info.dataType()) {
    auto castOpUp = std::make_unique<CastOp>(
        Onnx::Operators::Cast_9, t->info.dataType(), settings);
    CastOp *castOp = castOpUp.get();
    graph->moveIntoGraph(std::move(castOpUp));
    castOp->setName("Cast_" + offsetId);
    castOp->connectInTensor(CastOp::getInIndex(), offsetId);
    offsetId = graph->getIr().createIntermediateTensorId(offsetId);
    castOp->createAndConnectOutTensor(CastOp::getInIndex(), offsetId);
    castOp->setup();
    ops.push_back(castOp);
  }

  std::unique_ptr<AddOp> addOpUp =
      std::make_unique<AddOp>(Onnx::Operators::Add_7, settings);
  AddOp *addOp = addOpUp.get();
  graph->moveIntoGraph(std::move(addOpUp));
  addOp->setName("Add_" + tensorId);
  addOp->connectInTensor(AddOp::getArg0InIndex(), concatId);
  addOp->connectInTensor(AddOp::getArg1InIndex(), offsetId);
  addOp->createAndConnectOutTensor(AddOp::getOutIndex(), tensorId);
  addOp->setup();
  ops.push_back(addOp);

  return ops;
}

std::vector<Op *>
ShardingHelper::offsets(std::vector<TensorId> tensorIds,
                        TensorId concatId,
                        std::vector<Op::Settings> settings) const {
  if (settings.size() != tensorIds.size() && settings.size() != 1) {
    throw error("[ShardingHelper] Expected {} or 1 settings.",
                tensorIds.size());
  }

  std::vector<Op *> ops;

  auto t = graph->getIr().getTensor(concatId);

  for (int64_t b = 0; b < tensorIds.size(); ++b) {

    TensorId constId = createOrGetConstTensor<int>(
        t->info.dataType(), b, settings.at(b % settings.size()));

    auto newOps = offset(
        tensorIds.at(b), concatId, constId, settings.at(b % settings.size()));
    ops.insert(ops.end(), newOps.begin(), newOps.end());
  }
  return ops;
}

Tensor *ShardingHelper::initTensor(TensorInfo info,
                                   TensorId id,
                                   InitType type,
                                   Op::Settings settings) const {
  TensorId prefix = type == InitType::NoInit ? reservedConcatInitPrefix()
                                             : reservedInitPrefix();

  TensorId tensorId = prefix + id;

  auto initOpUp = std::make_unique<InitOp>(
      Onnx::CustomOperators::Init_1, info, TensorType::ActGrad, type, settings);
  Op *initOp = initOpUp.get();
  graph->moveIntoGraph(std::move(initOpUp));

  initOp->setName(tensorId);

  TensorId initId = graph->getIr().createIntermediateTensorId(tensorId);
  initOp->createAndConnectOutTensor(InitOp::getOutIndex(), initId);
  initOp->setup();

  return initOp->output->tensor(InitOp::getOutIndex());
}

Op *ShardingHelper::idLoss(ReductionType reductionType,
                           TensorId intermediateId,
                           TensorId lossOutId,
                           Op::Settings settings) const {
  // Do not include graph replication factor in the reduction
  auto idLossOpUp =
      std::make_unique<IdentityLossOp>(Onnx::AiGraphcore::OpSet1::IdentityLoss,
                                       reductionType,
                                       settings,
                                       ScaleByReplication::No);
  Op *idLossOp = idLossOpUp.get();
  graph->moveIntoGraph(std::move(idLossOpUp));

  idLossOp->connectInTensor(IdentityLossOp::getInIndex(), intermediateId);
  idLossOp->connectOutTensor(IdentityLossOp::getOutIndex(), lossOutId);
  idLossOp->setup();
  idLossOp->toLoss                 = PathToLoss::Yes;
  idLossOp->fromLoss               = PathFromLoss::Yes;
  idLossOp->settings.recomputeType = RecomputeType::Checkpoint;
  graph->getIr().setFinalLoss(lossOutId);
  return idLossOp;
}

Op *ShardingHelper::scale(float factor,
                          TensorId inId,
                          TensorId outId,
                          Op::Settings settings) const {
  auto scaleOpUp = std::make_unique<ScaleOp>(
      Onnx::AiGraphcore::OpSet1::Scale, factor, settings);
  Op *scaleOp = scaleOpUp.get();
  graph->moveIntoGraph(std::move(scaleOpUp));
  scaleOp->connectInTensor(ScaleOp::getInIndex(), inId);
  scaleOp->connectOutTensor(ScaleOp::getOutIndex(), outId);
  scaleOp->setup();
  return scaleOp;
}

TensorId ShardingHelper::createOrGetIndexTensor(uint32_t index,
                                                Op::Settings settings) const {
  std::stringstream ss;
  ss << reservedIndexPrefix() << std::to_string(index);
  ss << "_" << settings.tileSet;
  if (settings.vgraphId) {
    ss << "_" << *(settings.vgraphId);
  }

  TensorId id = graph->addScope(ss.str());
  if (!graph->getTensors().contains(id)) {
    TensorInfo indexTensorInfo(DataType::UINT32, {1});
    std::vector<uint32_t> idData(1, index);
    graph->getTensors().addConstInit(
        id, indexTensorInfo, reinterpret_cast<void *>(idData.data()));
  }
  return id;
}

template <class T>
TensorId ShardingHelper::createOrGetConstTensor(DataType type,
                                                T value,
                                                Op::Settings settings) const {
  std::stringstream ss;
  ss << reservedConstValuePrefix() << "_" << static_cast<int>(type) << "_"
     << std::to_string(value);
  ss << "_" << settings.tileSet;
  if (settings.vgraphId) {
    ss << "_" << *(settings.vgraphId);
  }
  TensorId id = graph->addScope(ss.str());
  if (!graph->getTensors().contains(id)) {
    TensorInfo indexTensorInfo(type, {1});
    std::vector<T> idData(1, value);
    graph->getTensors().addConstInit(
        id, indexTensorInfo, reinterpret_cast<void *>(idData.data()));
  }
  return id;
}

bool Op::canShard() const { return false; }

ReductionType Op::getShardReductionType(OutIndex index) const {
  return ReductionType::Sum;
}

ShardIdMap Op::shard(const ShardIdMap &inputs) {
  return Op::shard(
             ShardingPlan(ShardingMethod::DynamicShard, inputs, getGraph()))
      .getIdMap();
}

std::pair<ShardingPlan, ShardingPlan>
Op::adjustShardPlans(const ShardingPlan inputPlan) {
  auto &graph              = getGraph();
  auto &ir                 = graph.getIr();
  int64_t total_num_shards = inputPlan.getTotalNumShards();

  ShardingPlan adjustedInputPlan(inputPlan.getMethod(),
                                 inputPlan.getOpSettings());
  ShardingPlan outputPlan(inputPlan.getMethod());

  ShardingHelper helper(&graph);

  ShardIdMap inputIdMap     = inputPlan.getIdMap();
  ShardInfoMap inputInfoMap = inputPlan.getInfoMap();

  for (auto &tensorIdAndInfo : inputInfoMap) {
    int64_t num_shards = tensorIdAndInfo.second.infos.size();
    total_num_shards   = std::max(num_shards, total_num_shards);
  }
  adjustedInputPlan.setTotalNumShards(total_num_shards);
  outputPlan.setTotalNumShards(total_num_shards);

  for (auto &tensorIdAndInfo : inputInfoMap) {

    TensorId concatId    = tensorIdAndInfo.second.id;
    Shape origShape      = tensorIdAndInfo.second.info.shape();
    int64_t num_shards   = tensorIdAndInfo.second.infos.size();
    ShardTensorType type = tensorIdAndInfo.second.type;
    int64_t axis         = 0;

    if (num_shards > 0) {
      Shape sliceShape = tensorIdAndInfo.second.infos.front().shape();
      for (int64_t i = 0; i < sliceShape.size(); ++i) {
        if (sliceShape[i] * num_shards == origShape[i]) {
          axis = i;
        }
      }
    }

    // Prepare input tensors
    if (inputPlan.getMethod() == ShardingMethod::DynamicShard ||
        inputPlan.getMethod() == ShardingMethod::StaticShard) {
      auto it = inputIdMap.find(tensorIdAndInfo.first);
      if (it == inputIdMap.end()) {
        // Input was not sharded yet, but has a plan -> shard

        std::vector<TensorId> sliceIds;
        sliceIds.reserve(total_num_shards);

        std::vector<Op::Settings> sliceSettings(total_num_shards + 2, settings);
        for (int64_t b = 0; b < total_num_shards; ++b) {
          TensorId sliceId =
              ir.createSliceTensorId(concatId,
                                     static_cast<unsigned int>(b),
                                     static_cast<unsigned int>(b + 1));

          sliceIds.push_back(sliceId);
          if (inputPlan.getOpSettings().getShardSettings().size() >= b) {
            sliceSettings.at(b) =
                inputPlan.getOpSettings().getShardSettings().at(b);
          }
        }

        ShardIdMap map;
        map.insert({concatId, sliceIds});

        if (inputPlan.getOpSettings().hasPreSetting()) {
          sliceSettings.at(total_num_shards) =
              inputPlan.getOpSettings().getPreSetting();
          sliceSettings.at(total_num_shards + 1) =
              inputPlan.getOpSettings().getPreSetting();
        }

        for (size_t i = 0; i < sliceSettings.size(); ++i) {
          Tensor *t = getIr().getTensor(it->first);
          if (input->contains(t)) {
            auto indices = input->indices(t);
            sliceSettings.at(i) =
                adjustInSettings(indices.front(), sliceSettings.at(i));
          }
        }

        if (type == ShardTensorType::Slice) {
          if (inputPlan.getMethod() == ShardingMethod::DynamicShard) {
            helper.dynamicShard(axis, sliceIds, concatId, sliceSettings);
          } else {
            helper.staticShard(axis, sliceIds, concatId, sliceSettings);
          }
        } else if (type == ShardTensorType::Offset) {
          sliceSettings.resize(sliceIds.size(), Op::Settings(graph, ""));
          helper.offsets(sliceIds, concatId, sliceSettings);
        } else {
          throw error("[Op] Unsupported sharding tensor type.");
        }
        adjustedInputPlan.insertIdMap(map, graph);
        outputPlan.insertIdMap(map, graph);
      } else {
        // Input sharded with plan
        ShardIdMap map;
        map.insert({it->first, it->second});
        adjustedInputPlan.insertIdMap(map, graph);
      }
    } else if (inputPlan.getMethod() == ShardingMethod::Loop) {
      auto it = inputIdMap.find(tensorIdAndInfo.first);
      if (it == inputIdMap.end()) {
        // Input was not sharded yet, but has a plan
        ShardInfoMap map;
        map.insert(tensorIdAndInfo);
        adjustedInputPlan.insertInfoMap(map);
      } else {
        // Input sharded with plan -> concatenate again before loop

        std::vector<Op::Settings> concatSettings(total_num_shards + 2,
                                                 settings);
        for (int64_t b = 0; b < total_num_shards; ++b) {
          if (inputPlan.getOpSettings().getShardSettings().size() >= b) {
            concatSettings.at(b) =
                inputPlan.getOpSettings().getShardSettings().at(b);
          }
        }

        if (inputPlan.getOpSettings().hasPreSetting()) {
          concatSettings.at(total_num_shards) =
              inputPlan.getOpSettings().getPreSetting();
          concatSettings.at(total_num_shards + 1) =
              inputPlan.getOpSettings().getPreSetting();
        }

        for (size_t i = 0; i < concatSettings.size(); ++i) {
          Tensor *t = getIr().getTensor(it->first);
          if (input->contains(t)) {
            auto indices = input->indices(t);
            concatSettings.at(i) =
                adjustInSettings(indices.front(), concatSettings.at(i));
          }
        }

        TensorId newConcatId = ir.createIntermediateTensorId(concatId);

        ShardInfoMap map;
        // Info stays the same, but the newConcatId is connected
        map.insert({concatId,
                    {newConcatId,
                     tensorIdAndInfo.second.info,
                     tensorIdAndInfo.second.infos}});

        helper.dynamicConcat(axis, it->second, newConcatId, concatSettings);

        adjustedInputPlan.insertInfoMap(map);
        outputPlan.insertInfoMap(map);
      }
    }
  }

  return {adjustedInputPlan, outputPlan};
}

ShardingPlan Op::unrollShard(const ShardingPlan adjustedInputPlan,
                             ShardingPlan outputPlan) {
  auto &graph = getGraph();
  auto &ir    = graph.getIr();
  ShardingHelper helper(&graph);

  // Construct shards
  auto inputs = adjustedInputPlan.getIdMap();

  ShardIdMap shardOutputs;
  size_t num_shards = 1;
  for (auto &idkv : inputs) {
    num_shards = std::max(num_shards, idkv.second.size());
  }

  auto connectInTensorFn = [this](Op *op, InIndex index, TensorId tensorId) {
    IpuCopyOp *srcOp = dynamic_cast<IpuCopyOp *>(this);
    IpuCopyOp *dstOp = dynamic_cast<IpuCopyOp *>(op);
    if (srcOp && dstOp) {
      TensorId srcTensorId = srcOp->input->tensor(index)->id;
      dstOp->connectInTensor(index, tensorId, srcOp->getSourceIpu(srcTensorId));
    } else {
      op->connectInTensor(index, tensorId);
    }
  };

  std::vector<Op *> cloneOps;
  for (size_t b = 0; b < num_shards; ++b) {
    auto clonedOpUp = clone();
    auto cloneId    = graph.moveIntoGraph(std::move(clonedOpUp));
    Op *clonedOp    = graph.getOp(cloneId);
    clonedOp->disconnectAllInputs();
    clonedOp->disconnectAllOutputs();

    if (toLoss == PathToLoss::Yes && fromLoss == PathFromLoss::Yes) {
      clonedOp->fromLoss = PathFromLoss::No;
    }

    for (const auto &in : input->tensorMap()) {
      auto serializedTensor = inputs.find(in.second->id);
      if (serializedTensor == inputs.end()) {
        // Tensors not split
        connectInTensorFn(clonedOp, in.first, in.second->id);
      } else {
        if (serializedTensor->second.size() == num_shards) {
          // Tensors split dimension
          connectInTensorFn(clonedOp, in.first, serializedTensor->second[b]);
        } else if (serializedTensor->second.size() == 1) {
          // Tensors not split
          connectInTensorFn(clonedOp, in.first, serializedTensor->second[0]);
        } else {
          throw error("[Op::unrollShard] Number of input tensors must be 1 or "
                      "match the "
                      "serialziation factor {}",
                      num_shards);
        }
      }
    }
    for (const auto &out : output->tensorMap()) {
      TensorId sliceId =
          getIr().createSliceTensorId(out.second->id,
                                      static_cast<unsigned>(b),
                                      static_cast<unsigned>(b + 1));
      clonedOp->createAndConnectOutTensor(out.first, sliceId);
      shardOutputs[out.second->id].push_back(sliceId);
    }
    Settings cloneSettings = clonedOp->settings;
    if (adjustedInputPlan.getOpSettings().getShardSettings().size() > b) {
      cloneSettings =
          adjustedInputPlan.getOpSettings().getShardSettings().at(b);
    }
    configureShardedOp(clonedOp, &cloneSettings);
    clonedOp->setup();

    // Add rescaling if required
    auto cloneOutMap = clonedOp->output->tensorMap();
    for (const auto &out : cloneOutMap) {
      float factor = getShardRescaleFactor(clonedOp, out.first);

      if (factor != 1.0f) {
        TensorId tmpOutId = ir.createIntermediateTensorId(out.second->id);
        clonedOp->disconnectOutTensor(out.second);
        clonedOp->createAndConnectOutTensor(out.first, tmpOutId);
        clonedOp->setup();

        logging::op::trace("[Op::unrollShard] Adding rescaling on {}->{} by {}",
                           tmpOutId,
                           out.second->id,
                           factor);

        helper.scale(factor, tmpOutId, out.second->id, clonedOp->settings);
      }
    }

    logging::op::trace("[Op::unrollShard] Cloned op {} {} -> {}",
                       clonedOp->opid,
                       clonedOp->input->getIndexShapeMap(),
                       clonedOp->output->getIndexShapeMap());
  }

  for (auto out : shardOutputs) {
    TensorId oldOutId  = out.first;
    Tensor *oldOut     = graph.getTensors().get(oldOutId);
    auto reductionType = getShardReductionType(output->indices(oldOut).front());

    Tensor *newOut = graph.getTensors().get(shardOutputs.at(oldOutId).front());

    logging::op::trace(
        "[Op::unrollShard] {}; old output shape: {}, new output shape: {}x{}",
        debugName(),
        oldOut->info.shape(),
        shardOutputs.at(oldOutId).size(),
        newOut->info.shape());

    if (reductionType != ReductionType::NoReduction) {
      Settings postSettings =
          adjustedInputPlan.getOpSettings().hasPostSetting()
              ? adjustedInputPlan.getOpSettings().getPostSetting()
              : settings;

      if (output->contains(oldOut)) {
        auto indices = output->indices(oldOut);
        postSettings = adjustOutSettings(indices.front(), postSettings);
      }

      if (oldOut->info.nelms() == newOut->info.nelms() &&
          shardOutputs.at(oldOutId).size() > 1) {
        logging::op::trace(
            "[Op::unrollShard] {}; adding reduction over {} shards.",
            debugName(),
            shardOutputs.at(oldOutId).size());

        Op *reduceOp;

        switch (reductionType) {
        case ReductionType::Sum: {
          auto sumOpUp =
              std::make_unique<SumOp>(Onnx::Operators::Sum_8, postSettings);
          Op *sumOp = sumOpUp.get();
          graph.moveIntoGraph(std::move(sumOpUp));
          reduceOp = sumOp;
          break;
        }
        case ReductionType::Mean: {
          auto meanOpUp =
              std::make_unique<MeanOp>(Onnx::Operators::Mean_8, postSettings);
          Op *meanOp = meanOpUp.get();
          graph.moveIntoGraph(std::move(meanOpUp));
          reduceOp = meanOp;
          break;
        }
        case ReductionType::NoReduction:
          throw error(
              "[Op::unrollShard] Unsupported reduction type in shard() ({})",
              debugName());
          break;
        }
        cloneOps.push_back(reduceOp);

        disconnectOutTensor(oldOut);

        InIndex i = 0;
        for (auto shardOutId : shardOutputs[oldOutId]) {
          reduceOp->connectInTensor(i, shardOutId);
          ++i;
        }

        reduceOp->toLoss   = toLoss;
        reduceOp->fromLoss = fromLoss;
        if (toLoss == PathToLoss::Yes && fromLoss == PathFromLoss::Yes) {
          reduceOp->fromLoss = PathFromLoss::No;
          // New final loss
          auto tmpOutId = graph.getIr().createIntermediateTensorId(oldOutId);
          reduceOp->createAndConnectOutTensor(SumOp::getOutIndex(), tmpOutId);
          reduceOp->setup();
          helper.idLoss(reductionType, tmpOutId, oldOutId, postSettings);
        } else {
          reduceOp->connectOutTensor(SumOp::getOutIndex(), oldOutId);
          reduceOp->setup();
        }

        shardOutputs[oldOutId] = {oldOutId};
      }
    }
    outputPlan.insertIdMap(shardOutputs, graph);
  }

  graph.topoCons->transferToMultiple(this, cloneOps);

  return outputPlan;
}

ShardingPlan Op::loopShard(const ShardingPlan adjustedInputPlan,
                           ShardingPlan outputPlan) {
  auto &graph = getGraph();
  auto &ir    = graph.getIr();
  ShardingHelper helper(&graph);

  std::vector<Op *> shardedOps;

  // Construct loop

  Settings loopSettings =
      adjustedInputPlan.getOpSettings().getShardSettings().size() > 0
          ? adjustedInputPlan.getOpSettings().getShardSettings().at(0)
          : settings;
  Settings insideLoopSettings = loopSettings;

  auto subgraphId    = ir.createUniqueSubgraphId({"loop"});
  auto &subgraph     = ir.createGraph(subgraphId);
  auto subgraphScope = subgraph.getScope();

  // Ops inside the loop must use the correct subgraph
  insideLoopSettings.graph            = subgraph;
  insideLoopSettings.executionContext = ExecutionContext::Subgraph;
  insideLoopSettings.scope            = subgraphScope;

  // Iterate over Op outputs and inputs
  auto outputMap = output->tensorMap();
  auto inputMap  = input->tensorMap();

  ShardingHelper subgraphHelper(&subgraph);

  auto loopOpUp = std::make_unique<LoopOp>(
      Onnx::Operators::Loop_11, loopSettings, subgraph);
  LoopOp *loopOp = loopOpUp.get();
  graph.moveIntoGraph(std::move(loopOpUp));
  loopOp->setTripCountValue(adjustedInputPlan.getTotalNumShards());
  shardedOps.push_back(loopOp);
  // Extra outline attributes not relevant on the LoopOp
  loopOp->settings.extraOutlineAttributes.clear();

  // Move loop-sharded Op into subgraph
  auto clonedOpUp = clone();
  auto cloneId    = subgraph.moveIntoGraph(std::move(clonedOpUp));
  Op *clonedOp    = subgraph.getOp(cloneId);
  clonedOp->disconnectAllInputs();
  clonedOp->disconnectAllOutputs();

  bool requiresIdLoss = false;
  // Adjust from/to loss
  loopOp->toLoss   = toLoss;
  loopOp->fromLoss = fromLoss;
  if (toLoss == PathToLoss::Yes && fromLoss == PathFromLoss::Yes) {
    loopOp->fromLoss   = PathFromLoss::No;
    clonedOp->fromLoss = PathFromLoss::No;
    requiresIdLoss     = true;
  }

  // Add mandatory loop iterator tensor to subgraph (is not an output)
  TensorId loopItScopedId = subgraph.addScope(reservedLoopIteratorPrefix());
  subgraph.addInput(loopItScopedId, TensorInfo(DataType::INT32, {}));

  // Add mandatory loop condition tensor to subgraph (is also an output)
  TensorId loopCondScopedId = subgraph.addScope(reservedLoopCondPrefix());
  subgraph.addInput(loopCondScopedId, TensorInfo(DataType::BOOL, {}));
  subgraph.markAsOutput(loopCondScopedId);

  // Explicit input/output pairs
  InIndex loopInIndex  = LoopOp::getFirstInputInIndex();
  InIndex loopOutIndex = 0;

  // Index / iterators (one per required VGID and TileSet)
  std::set<TensorId> serialIndexTensorScopedIds;
  auto getOrCreateIterator = [&serialIndexTensorScopedIds,
                              &loopOp,
                              &loopInIndex,
                              &loopOutIndex,
                              &subgraph,
                              &helper,
                              &subgraphHelper,
                              &ir](Settings indexSettings) {
    // Connect a loop iterator starting at 0
    TensorId serialIndexTensorId =
        helper.createOrGetIndexTensor(0, indexSettings);
    auto serialIndexTensorScopedId = subgraph.addScope(serialIndexTensorId);

    if (serialIndexTensorScopedIds.find(serialIndexTensorScopedId) ==
        serialIndexTensorScopedIds.end()) {
      serialIndexTensorScopedIds.insert(serialIndexTensorScopedId);

      loopOp->addLoopInput(
          loopInIndex++, serialIndexTensorId, serialIndexTensorScopedId, false);

      // Increment the loop index inside of the loop
      std::unique_ptr<AddOp> indexAddOpUp =
          std::make_unique<AddOp>(Onnx::Operators::Add_7, indexSettings);
      Op *indexAddOp = indexAddOpUp.get();
      subgraph.moveIntoGraph(std::move(indexAddOpUp));
      // Increment by 1
      TensorId constId = subgraphHelper.createOrGetConstTensor<uint32_t>(
          DataType::UINT32, 1, indexSettings);
      indexAddOp->connectInTensor(AddOp::getArg0InIndex(),
                                  serialIndexTensorScopedId);
      indexAddOp->connectInTensor(AddOp::getArg1InIndex(), constId);
      TensorId updatedSerialIndexTensorScopedId =
          ir.createIntermediateTensorId(serialIndexTensorScopedId);
      TensorId updatedSerialIndexTensorId =
          subgraph.removeScope(updatedSerialIndexTensorScopedId);
      indexAddOp->createAndConnectOutTensor(AddOp::getOutIndex(),
                                            updatedSerialIndexTensorScopedId);
      indexAddOp->setup();
      // Extra outline attributes not relevant on the indexAddOp
      indexAddOp->settings.extraOutlineAttributes.clear();
      loopOp->addLoopOutput(loopOutIndex++,
                            updatedSerialIndexTensorId,
                            updatedSerialIndexTensorScopedId,
                            false);
    }
    return serialIndexTensorScopedId;
  };

  // Connect Op inputs as implicit loop inputs
  for (const auto &in : inputMap) {
    TensorId opInId;
    auto infoMap = adjustedInputPlan.getInfoMap();

    auto inScopedId = subgraph.addScope(in.second->id);

    Settings preInsideLoopSettings =
        adjustInSettings(in.first, insideLoopSettings);

    auto it = infoMap.find(in.second->id);
    if (it != infoMap.end()) {
      // Connect from shard plan
      loopOp->addLoopInput(std::max(LoopOp::getFirstInputInIndex(),
                                    loopOp->input->maxIndex() + 1),
                           it->second.id,
                           inScopedId,
                           false);

      Shape origShape        = it->second.info.shape();
      TensorId sliceScopedId = ir.createIntermediateTensorId(inScopedId);

      if (it->second.type == ShardTensorType::Slice) {
        Shape sliceShape   = it->second.infos.front().shape();
        int64_t num_shards = it->second.infos.size();
        int64_t axis       = 0;

        for (int64_t i = 0; i < sliceShape.size(); ++i) {
          if (sliceShape[i] * num_shards == origShape[i]) {
            axis = i;
          }
        }

        subgraphHelper.dynamicSlice(axis,
                                    num_shards,
                                    sliceScopedId,
                                    inScopedId,
                                    getOrCreateIterator(preInsideLoopSettings),
                                    preInsideLoopSettings);
      } else if (it->second.type == ShardTensorType::Offset) {
        subgraphHelper.offset(sliceScopedId,
                              inScopedId,
                              getOrCreateIterator(preInsideLoopSettings),
                              preInsideLoopSettings);

      } else {
        throw error("[Op::loopShard] Unsupported sharding tensor type.");
      }

      opInId = sliceScopedId;
    } else {
      // Connect directly
      loopOp->addLoopInput(std::max(LoopOp::getFirstInputInIndex(),
                                    loopOp->input->maxIndex() + 1),
                           in.second->id,
                           inScopedId,
                           false);
      opInId = inScopedId;
    }

    // Connect the cloned Op inputs
    clonedOp->connectInTensor(in.first, opInId);
  }

  // Connect the cloned Op outputs
  for (const auto &out : outputMap) {
    TensorId shardOutId = subgraph.addScope(out.second->id);
    clonedOp->createAndConnectOutTensor(out.first, shardOutId);
  }

  // Setup the cloned Op (calculates the proper output shapes)
  configureShardedOp(clonedOp, &insideLoopSettings);
  clonedOp->setup();

  // Add rescaling if required
  auto cloneOutMap = clonedOp->output->tensorMap();
  for (const auto &out : cloneOutMap) {
    float factor = getShardRescaleFactor(clonedOp, out.first);

    if (factor != 1.0f) {
      TensorId tmpOutId = ir.createIntermediateTensorId(out.second->id);
      clonedOp->disconnectOutTensor(out.second);
      clonedOp->createAndConnectOutTensor(out.first, tmpOutId);
      clonedOp->setup();

      logging::op::trace("[Op::loopShard] Adding rescaling on {}->{} by {}",
                         tmpOutId,
                         out.second->id,
                         factor);

      subgraphHelper.scale(
          factor, tmpOutId, out.second->id, insideLoopSettings);
    }
  }

  // Add loop outputs and explicit inputs, add dynamic updates &
  // accumulation
  std::map<TensorId, ReductionType> outReductionMap;
  for (const auto &out : outputMap) {
    Shape outShape   = out.second->info.shape();
    Shape sliceShape = cloneOutMap.at(out.first)->info.shape();
    TensorId sliceId = cloneOutMap.at(out.first)->id;

    bool reduce = outShape == sliceShape;

    Settings initSettings = adjustOutSettings(
        out.first,
        adjustedInputPlan.getOpSettings().hasPreSetting()
            ? adjustedInputPlan.getOpSettings().getPreSetting()
            : settings);

    // Explicit input -> explicit output
    Tensor *initTensor =
        helper.initTensor(out.second->info,
                          out.second->id,
                          reduce ? InitType::Zero : InitType::NoInit,
                          initSettings);

    Settings postInsideLoopSettings =
        adjustOutSettings(out.first, insideLoopSettings);

    TensorId serialIndexTensorScopedId;
    if (!reduce) {
      serialIndexTensorScopedId = getOrCreateIterator(postInsideLoopSettings);
    }

    auto initScopedId = subgraph.addScope(initTensor->id);
    loopOp->addLoopInput(loopInIndex++, initTensor->id, initScopedId, false);
    TensorId updatedTensorId = ir.createIntermediateTensorId(initScopedId);

    if (reduce) {
      ReductionType reductionType = getShardReductionType(out.first);
      outReductionMap.insert({out.second->id, reductionType});
      if (reductionType == ReductionType::Mean ||
          reductionType == ReductionType::Sum) {
        auto addOpUp = std::make_unique<AddOp>(Onnx::Operators::Add_7,
                                               postInsideLoopSettings);
        Op *addOp    = addOpUp.get();
        subgraph.moveIntoGraph(std::move(addOpUp));
        addOp->connectInTensor(AddOp::getArg0InIndex(), initScopedId);
        addOp->connectInTensor(AddOp::getArg1InIndex(), sliceId);
        addOp->createAndConnectOutTensor(AddOp::getOutIndex(), updatedTensorId);
        addOp->setup();
      } else {
        throw error(
            "[Op::loopShard] Unsupported reduction type in shard() ({})",
            debugName());
        break;
      }
      // Reduced outputs aren't shardable, and are therefore not added
      // to the output plan
    } else {
      outReductionMap.insert({out.second->id, ReductionType::NoReduction});
      int64_t num_shards = adjustedInputPlan.getTotalNumShards();
      int64_t axis       = 0;

      for (int64_t i = 0; i < sliceShape.size(); ++i) {
        if (sliceShape[i] * num_shards == outShape[i]) {
          axis = i;
        }
      }

      logging::op::trace("[Op::loopShard] Dynamic update {} {} -> {} {} (num "
                         "shards: {}, axis: {})",
                         sliceId,
                         sliceShape,
                         updatedTensorId,
                         outShape,
                         num_shards,
                         axis);

      subgraphHelper.dynamicUpdate(axis,
                                   num_shards,
                                   sliceId,
                                   initScopedId,
                                   updatedTensorId,
                                   serialIndexTensorScopedId,
                                   postInsideLoopSettings);

      // Add shardable output to the output plan
      ShardInfoMap map;
      std::vector<TensorInfo> outShardInfo(
          num_shards, clonedOp->outTensor(out.first)->info);
      map.insert(
          {out.second->id, {out.second->id, out.second->info, outShardInfo}});
      outputPlan.insertInfoMap(map);
    }
    loopOp->addLoopOutput(
        loopOutIndex++, out.second->id, updatedTensorId, false);
  }

  // Setup the loop
  loopOp->setup();

  // Post-processing outputs after the loop
  for (const auto &out : outputMap) {

    // Add division for mean reduction, if required
    auto outReduction = outReductionMap.at(out.second->id);
    if (outReduction == ReductionType::Mean) {
      auto tmpOutId = graph.getIr().createIntermediateTensorId(out.second->id);

      Op *prod          = out.second->getProducer();
      OutIndex outIndex = prod->output->indices(out.second).front();
      prod->disconnectOutTensor(out.second);
      prod->createAndConnectOutTensor(outIndex, tmpOutId);
      prod->setup();

      float factor =
          1.f / static_cast<float>(adjustedInputPlan.getTotalNumShards());
      logging::op::trace(
          "[Op::loopShard] Adding ReductionType::{} scaling on {}->{} by {}",
          outReduction,
          tmpOutId,
          out.second->id,
          factor);
      auto scaleOp =
          helper.scale(factor,
                       tmpOutId,
                       out.second->id,
                       adjustedInputPlan.getOpSettings().hasPostSetting()
                           ? adjustedInputPlan.getOpSettings().getPostSetting()
                           : settings);
      shardedOps.push_back(scaleOp);
    }

    // Add identity loss, if required
    if (requiresIdLoss && out.second->id == ir.getFinalLossId()) {
      Settings postSettings =
          adjustedInputPlan.getOpSettings().hasPostSetting()
              ? adjustedInputPlan.getOpSettings().getPostSetting()
              : settings;

      postSettings = adjustOutSettings(out.first, postSettings);

      auto tmpOutId = graph.getIr().createIntermediateTensorId(out.second->id);

      Op *prod          = out.second->getProducer();
      OutIndex outIndex = prod->output->indices(out.second).front();
      prod->disconnectOutTensor(out.second);
      prod->createAndConnectOutTensor(outIndex, tmpOutId);
      prod->setup();

      Op *idLossOp = helper.idLoss(outReductionMap.at(out.second->id),
                                   tmpOutId,
                                   out.second->id,
                                   postSettings);
      shardedOps.push_back(idLossOp);
    }
  }

  graph.topoCons->transferToMultiple(this, shardedOps);

  return outputPlan;
}

ShardingPlan Op::shard(const ShardingPlan inputPlan) {
  auto &graph = getGraph();
  ShardingHelper helper(&graph);

  auto plans                     = adjustShardPlans(inputPlan);
  ShardingPlan adjustedInputPlan = plans.first;
  ShardingPlan outputPlan        = plans.second;

  // Shard operation
  if (adjustedInputPlan.getMethod() == ShardingMethod::DynamicShard ||
      adjustedInputPlan.getMethod() == ShardingMethod::StaticShard) {
    outputPlan = unrollShard(adjustedInputPlan, outputPlan);
  } else if (adjustedInputPlan.getMethod() == ShardingMethod::Loop) {
    outputPlan = loopShard(adjustedInputPlan, outputPlan);
  }

  return outputPlan;
}

void Op::configureShardedOp(Op *const shardOp,
                            const Settings *const settings_) const {
  if (settings_) {
    shardOp->settings = *settings_;
  }
}

} // namespace popart
