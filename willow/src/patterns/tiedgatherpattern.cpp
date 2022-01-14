// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <popart/patterns/tiedgatherpattern.hpp>

#include <patterns/tiedgatherutils/tgutils.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op.hpp>
#include <popart/op/accumulate.hpp>
#include <popart/op/adamupdater.hpp>
#include <popart/op/adamvarupdate.hpp>
#include <popart/op/add.hpp>
#include <popart/op/collectives/replicatedallgather.hpp>
#include <popart/op/collectives/replicatedreducescatter.hpp>
#include <popart/op/detach.hpp>
#include <popart/op/div.hpp>
#include <popart/op/dropout.hpp>
#include <popart/op/gather.hpp>
#include <popart/op/matmul.hpp>
#include <popart/op/mul.hpp>
#include <popart/op/reshape.hpp>
#include <popart/op/slice.hpp>
#include <popart/op/subtract.hpp>
#include <popart/op/tiedgather.hpp>
#include <popart/op/transpose.hpp>
#include <popart/opidentifier.hpp>
#include <popart/patterns/patterns.hpp>
#include <popart/topocons.hpp>

#include <queue>
#include <vector>

namespace popart {

using SerialiseSettings = MatMulBaseOp::SerialiseSettings;

namespace {
bool canSerialiseTiedGather(const MatMulOp *matmul);
bool exactlyOneProducedByTranspose(const MatMulOp *matmul,
                                   const GatherOp *gather);
MatMulOp *findMatMulThatConsumesSameRootWeight(const GatherOp *gather) {
  return tgutil::weightConsumedBy<MatMulOp>(
      gather->input->tensor(GatherOp::dataInIndex()));
}
} // namespace

bool TiedGatherPattern::matches(Op *op) const {
  const auto &ir = op->getIr();
  // Only run in the fwd pass
  if (ir.hasConstructedBackwards()) {
    return false;
  }
  if (ir.isTraining() && !ir.getSessionOptions().enableGradientAccumulation) {
    return false;
  }
  if (op->isConvertibleTo<GatherOp>() && !op->isConvertibleTo<TiedGatherOp>()) {
    auto gather = dynamic_cast<GatherOp *>(op);
    auto matmul = findMatMulThatConsumesSameRootWeight(gather);

    // For the tile layouts to be favourable, we need a Gather whose axis is
    // the output channel dimension of the MatMul. One way to achieve this is
    // to transpose the weight and gather on axis 0. There are other
    // possibilities too, but this is the only one this pattern has so far
    // been validated for.
    const bool gatherAxisIsMatMulOutChannelDim = gather->getAxis() == 0;

    if (matmul && gatherAxisIsMatMulOutChannelDim &&
        canSerialiseTiedGather(matmul) &&
        exactlyOneProducedByTranspose(matmul, gather)) {
      return true;
    }
  }
  return false;
}

std::vector<const Tensor *> TiedGatherPattern::touches(Op *) const {
  return {};
}

bool TiedGatherPattern::apply(Op *op) const {
  logging::pattern::trace("[TiedGatherPattern] Applying to Op {}",
                          op->debugName());

  auto &graph = op->getGraph();

  auto gather = dynamic_cast<GatherOp *>(op);
  auto matmul = findMatMulThatConsumesSameRootWeight(gather);
  if (!matmul) {
    throw internal_error(
        "[TiedGatherPattern] Can not find corresponding MatMul for GatherOp "
        "`{}`. Pattern should not have matched.",
        gather->debugName());
  }

  // (1)
  matmul->setUseFullyConnectedPass(false);

  const auto axis          = gather->getAxis();
  const auto availMemProp  = gather->getAvailableMemoryProportion();
  const auto serialisation = matmul->getSerialiseSettings();

  auto data          = gather->input->tensor(GatherOp::dataInIndex());
  const auto indices = gather->input->tensor(GatherOp::indicesInIndex());
  const auto out     = gather->output->tensor(GatherOp::outIndex());

  // Disconnect "out" so it can be connected to the replacing ops.
  gather->disconnectAllOutputs();

  const std::string &name =
      gather->name().empty() ? std::to_string(gather->id) : gather->name();

  // (2)
  auto detach =
      graph.createOp<DetachOp>(Onnx::CustomOperators::Detach_1,
                               Op::Settings(graph, name + "/TiedGatherDetach"));
  transferBaseProperties(gather, detach);
  detach->connectInTensor(DetachOp::getInIndex(), data->id);
  auto detached_data_id = data->id + "/detached";
  detach->createAndConnectOutTensor(DetachOp::getOutIndex(), detached_data_id);
  detach->setup();
  data = graph.getTensors().get(detached_data_id);

  auto replaceWithTiedGather = [&](const TensorId &dict,
                                   const TensorId &ind,
                                   int64_t i,
                                   const std::string &debugPrefix) {
    bool zeroOutOfRangeIndices = true;
    auto tiedGather =
        graph.createOp<TiedGatherOp>(axis,
                                     Op::Settings(graph, debugPrefix),
                                     availMemProp,
                                     zeroOutOfRangeIndices);
    transferBaseProperties(gather, tiedGather);

    tiedGather->connectInTensor(TiedGatherOp::dataInIndex(), dict);
    tiedGather->connectInTensor(TiedGatherOp::indicesInIndex(), ind);

    auto outId = out->id;
    if (i >= 0) {
      outId = debugPrefix + ":0";
      tiedGather->createAndConnectOutTensor(TiedGatherOp::outIndex(), outId);
    } else {
      tiedGather->connectOutTensor(TiedGatherOp::outIndex(), outId);
    }

    graph.topoCons->transfer(gather, tiedGather);

    tiedGather->setup();

    return outId;
  };

  if (serialisation.factor <= 1 ||
      serialisation.mode == SerialiseSettings::Mode::None) {
    // (3)
    replaceWithTiedGather(data->id, indices->id, -1, name);
  } else {
    // (4)
    if (serialisation.mode != SerialiseSettings::Mode::OutputChannels) {
      throw internal_error(
          "[TiedGatherPattern] Unsupported matmul serialisation settings. Only "
          "no matmul serialisation or mode "
          "MatMulBaseOp::SerialisationSettings::Mode::OutputChannels are "
          "supported. TiedGatherPattern::matches should have returned false.");
    }

    auto insertSliceOp = [&](int64_t starts,
                             int64_t ends,
                             const std::string &debugPrefix) {
      auto slice =
          graph.createOp<SliceOp>(Onnx::AiOnnx::OpSet9::Slice,
                                  std::vector<int64_t>({starts}),
                                  std::vector<int64_t>({ends}),
                                  std::vector<int64_t>({axis}),
                                  Op::Settings(graph, debugPrefix + "/slice"));
      transferBaseProperties(gather, slice);
      slice->connectInTensor(SliceOp::getInIndex(), data->id);
      auto data_slice = debugPrefix + "/slice:0";
      slice->createAndConnectOutTensor(SliceOp::getOutIndex(), data_slice);
      slice->setup();
      return data_slice;
    };

    int sub_i_                = 0; // Counter used in lambda.
    auto subtractWithConstant = [&](Tensor *a,
                                    int64_t c,
                                    const std::string &debugPrefix) {
      auto sub = graph.createOp<SubtractOp>(
          Onnx::Operators::Sub_7, Op::Settings(graph, debugPrefix + "/sub"));
      transferBaseProperties(gather, sub);
      sub->connectInTensor(SubtractOp::getArg0InIndex(), a->id);
      // Create constant to subtract from
      auto subConstId =
          debugPrefix + "/" + a->id + "_sub_const_" + std::to_string(sub_i_++);
      TensorInfo subInfo(a->info.dataType(), {1});
      std::vector<unsigned> d(1, c);
      graph.getTensors().addConstInit(subConstId, subInfo, d.data());
      sub->connectInTensor(SubtractOp::getArg1InIndex(), subConstId);
      auto indicesSub = debugPrefix + "/sub:0";
      sub->createAndConnectOutTensor(SubtractOp::getOutIndex(), indicesSub);
      sub->setup();
      return indicesSub;
    };

    auto insertAddOp = [&](const TensorId &a,
                           const TensorId &b,
                           const TensorId &out,
                           const std::string &debugPrefix) {
      auto addOp = graph.createOp<AddOp>(
          Onnx::Operators::Add_6, Op::Settings(graph, debugPrefix + "/add"));
      transferBaseProperties(gather, addOp);
      addOp->connectInTensor(AddOp::getArg0InIndex(), a);
      addOp->connectInTensor(AddOp::getArg1InIndex(), b);
      if (graph.getTensors().contains(out)) {
        addOp->connectOutTensor(AddOp::getOutIndex(), out);
      } else {
        addOp->createAndConnectOutTensor(AddOp::getOutIndex(), out);
      }
      addOp->setup();
      return out;
    };

    TensorId tmpId;
    for (int64_t i = 0; i < serialisation.factor; i++) {
      int64_t sliceSize = data->info.dim(axis) / serialisation.factor;
      auto serialName   = name + "/" + std::to_string(i);
      // Slice the Dictionary
      auto dataSlice =
          insertSliceOp(i * sliceSize, (i + 1) * sliceSize, serialName);
      // Subtract the indices
      auto indicesSub =
          subtractWithConstant(indices, i * sliceSize, serialName);
      // Add the tied gather to the graph
      auto nextId = replaceWithTiedGather(dataSlice, indicesSub, i, serialName);

      // Add the results
      if (i == 0) {
        tmpId = nextId;
      } else {
        auto outId = out->id;
        if (i < serialisation.factor - 1) {
          outId += "_tmp" + std::to_string(i);
        }
        tmpId = insertAddOp(tmpId, nextId, outId, serialName);

        // Tie the add to happen directly after the gather
        graph.topoCons->insert(graph.getTensors().get(nextId)->getProducer(),
                               graph.getTensors().get(tmpId)->getProducer(),
                               true);
      }
    }
  }

  gather->disconnectAllInputs();
  graph.eraseOp(gather->id);

  return true;
}

namespace {
PatternCreator<TiedGatherPattern> tiedGatherer("TiedGather",
                                               false,  // Off by default
                                               false); // Not mandatory
} // namespace

/****** TiedGatherAccumulate ******/

bool TiedGatherAccumulatePattern::matches(Op *op) const {
  // Only works with gradient accumulation
  if (!op->getIr().getSessionOptions().enableGradientAccumulation) {
    return false;
  }
  // Only run after the optimizers have been created
  if (!op->getIr().hasDecomposedOptimizers()) {
    return false;
  }
  return op->isConvertibleTo<TiedGatherGradOp>();
}

std::vector<const Tensor *> TiedGatherAccumulatePattern::touches(Op *) const {
  return {};
}

/**
  This pattern matches for graphs of the shape.

        Weight
            |              \
  Grad - TiedGatherGrad   MatMul
                            |
                Accum  -  Accumulate

  And will perform the following transformation
    1) Replace TiedGatherGrad with SparseAccumulate

  Resulting in:

        Weight
          |              \
          |             MatMul
          |               |
  Grad   Accum  -  Accumulate
    |     |               |
    SparseAccumulate  - Optimizer

  (--> is a topocon)
*/
bool TiedGatherAccumulatePattern::apply(Op *op) const {
  logging::pattern::trace("[TiedGatherAccumulatePattern] Applying to Op {}",
                          op->debugName());

  auto gatherGrad = dynamic_cast<TiedGatherGradOp *>(op);
  auto gather     = gatherGrad->fwdOp;
  auto rootWeight =
      tgutil::getVariable(gather->input->tensor(GatherOp::dataInIndex()));

  auto gatherOps = tgutil::findAllConsumers<TiedGatherOp>(rootWeight);

  // Get all the Accumulate ops in the normal context
  std::vector<AccumulateOp *> accumOps;

  auto updateOps =
      tgutil::findAllConsumers<VarUpdateWithUpdaterOp,
                               ExecutionContext::AccumulateOuterFragment>(
          rootWeight);
  if (updateOps.size() < 1) {
    // Optimiser decompose pattern has not yet run.
    throw internal_error(
        "[TiedGatherAccumulatePattern] Could not find var update ops for "
        "weight {}. Either matches should have returned false or the "
        "search algorithm could not find them.",
        rootWeight->id);
  }

  for (size_t i = 0; i < updateOps.size(); i++) {
    auto varUpdateOp = updateOps[i];

    auto accum =
        varUpdateOp->inTensor(VarUpdateWithUpdaterOp::getUpdaterInIndex());
    // Accumulate Ops in the normal fragment are Gradient Accumulation.
    auto accumOp =
        tgutil::searchProducersFor<AccumulateOp, ExecutionContext::Normal>(
            accum);

    if (accumOp) {
      auto exists =
          std::find_if(accumOps.begin(), accumOps.end(), [&accumOp](Op *op) {
            return op->id == accumOp->id;
          });
      if (exists == accumOps.end()) {
        accumOps.push_back(accumOp);
      }
    } else {
      logging::pattern::trace("[TiedGatherAccumulatePattern] Could not "
                              "find outer gradient accumulation AccumulateOp "
                              "in producers of accumulator tensor {}.",
                              accum->id);
    }
  }

  if (accumOps.size() != gatherOps.size()) {
    // If matches has returned true (meaning TiedGatherPattern must have run),
    // there will be no correctly wired weight update due to the detach op
    // inserted in the forward graph. This pattern needs the above condition to
    // work. Therefore, if we cannot continue here, the error is irrecoverable.
    throw internal_error(
        "[TiedGatherAccumulatePattern] Could only find {} accumulate ops, but "
        "there are {} gather ops. Perhaps something else in PopART has changed "
        "and now the helper functions that search for these ops need updating.",
        accumOps.size(),
        gatherOps.size());
  }

  // Match up the GatherOps with the corresponding AccumulateOps from their
  // weight's update step.
  // TODO(T42654): Find a more robust way than sorting input ids.
  std::sort(accumOps.begin(), accumOps.end(), [](const Op *l, const Op *r) {
    return l->input->tensor(AccumulateOp::getVarToUpdateInIndex())
               ->id.compare(
                   r->input->tensor(AccumulateOp::getVarToUpdateInIndex())
                       ->id) < 0;
  });
  std::sort(gatherOps.begin(), gatherOps.end(), [](const Op *l, const Op *r) {
    return l->name().compare(r->name()) < 0;
  });

  auto itr = std::find(gatherOps.begin(), gatherOps.end(), gather);
  if (itr == gatherOps.end()) {
    throw internal_error(
        "[TiedGatherAccumulatePattern] Could not find the GatherOp this "
        "pattern has matched for `{}` in the consumers of the root weight "
        "we found for it `{}`. Perhaps something else in PopART has changed "
        "and now the helper functions that search for these ops need updating.",
        gather->name(),
        rootWeight->id);
  }

  unsigned serialIndex = std::distance(gatherOps.begin(), itr);

  auto denseAccumOp = accumOps[serialIndex];

  auto accumId  = denseAccumOp->inId(AccumulateOp::getVarToUpdateInIndex());
  auto weightId = gather->inId(GatherOp::dataInIndex());
  logging::pattern::trace(
      "[TiedGatherAccumulatePattern] Using tied accumulator {} for {}",
      accumId,
      gather->name());

  // TiedGatherPattern will have added a DetachOp and possibly a SliceOp
  // between the gather and root weight.
  auto gather_data = tgutil::maybeTraverseProducer<DetachOp>(
      DetachOp::getInIndex(),
      tgutil::maybeTraverseProducer<BaseSliceOp>(
          BaseSliceOp::getInIndex(),
          gather->input->tensor(GatherOp::dataInIndex())));
  if (tgutil::isProducedByTranspose(gather_data)) {
    // Transpose must be inplace so the accumulator is actually updated
    accumId = inplaceTranspose(accumId, gatherGrad);
  }

  auto &graph = op->getGraph();

  auto accumType = denseAccumOp->getAccumulationType();
  Tensor *factor =
      denseAccumOp->getFactor().isConst()
          ? nullptr
          : denseAccumOp->inTensor(SparseAccumulateOp::getFactorInIndex());

  if (factor != nullptr && accumType == AccumulationType::Mean) {
    auto inv_counter = factor->id + "_inverse";
    if (!graph.getTensors().contains(inv_counter)) {
      TensorInfo one_info(factor->info.dataType(), {});
      std::vector<float> one_data(one_info.nelms(), 1);
      const auto &one_id = graph.getIr().createIntermediateTensorId("one");
      graph.getTensors().addConstInit(one_id, one_info, one_data.data());
      auto inv_op = graph.createConnectedOp<DivOp>(
          {{DivOp::getArg0InIndex(), one_id},
           {DivOp::getArg1InIndex(), factor->id}},
          {{DivOp::getOutIndex(), inv_counter}},
          Onnx::Operators::Div_7,
          Op::Settings(graph, "mean_accumulate_inverse"));
      transferBaseProperties(gatherGrad, inv_op);

      for (auto cons : factor->consumers.getOps()) {
        if (cons->isConvertibleTo<AccumulateOp>() &&
            cons->inId(AccumulateOp::getVarToUpdateInIndex()) == factor->id) {
          graph.topoCons->insert(cons, inv_op);
        }
      }
    }
    accumType = AccumulationType::DampenedAdd;
    factor    = graph.getTensor(inv_counter);
  }

  auto sparseAccumOp = graph.createOp<SparseAccumulateOp>(
      accumType,
      denseAccumOp->getFactor(),
      gatherGrad->getAxis(),
      Op::Settings(graph, "_tiedAccumulate/" + std::to_string(serialIndex)));
  transferBaseProperties(gatherGrad, sparseAccumOp);

  // Inputs

  // Accumulator
  sparseAccumOp->connectInTensor(SparseAccumulateOp::getVarToUpdateInIndex(),
                                 accumId);
  // Gradients
  sparseAccumOp->connectInTensor(SparseAccumulateOp::getUpdaterInIndex(),
                                 gatherGrad->inId(GatherGradOp::gradInIndex()));
  // Scale
  if (!denseAccumOp->getFactor().isConst()) {
    sparseAccumOp->connectInTensor(
        // the index at which the dampening scale factor is received,
        SparseAccumulateOp::getFactorInIndex(),
        // the name of the dampening scale factor
        factor->id);
  }
  // Indices
  sparseAccumOp->connectInTensor(
      SparseAccumulateOp::getIndicesInIndex(),
      gatherGrad->inId(GatherGradOp::indicesInIndex()));

  // Original weight to be cloned
  sparseAccumOp->connectInTensor(
      SparseAccumulateOp::getOriginalVarToUpdateInIndex(), weightId);

  // Transfer TopoCons
  graph.topoCons->transfer(gatherGrad, sparseAccumOp);

  // gatherGrad output that will be isolated
  auto gatherGradId = gatherGrad->outId(TiedGatherGradOp::gradOutIndex());

  // Remove TiedGatherGrad
  gatherGrad->disconnectAllInputs();
  gatherGrad->disconnectAllOutputs();
  graph.eraseOp(gatherGrad->id);

  // Outputs
  sparseAccumOp->createAndConnectOutTensor(
      SparseAccumulateOp::getUpdatedVarOutIndex(),
      sparseAccumOp->name() + ":0");

  // remove the gatherGrad output
  graph.getTensors().remove(gatherGradId);

  // Finalise sparse op
  sparseAccumOp->setup();

  return true;
}

TensorId TiedGatherAccumulatePattern::inplaceTranspose(TensorId tid,
                                                       Op *op) const {
  auto &graph = op->getGraph();

  // TransposeInplaceOp's constructor requires a transposeOp
  auto outplace_up =
      std::make_unique<TransposeOp>(Onnx::AiOnnx::OpSet9::Transpose,
                                    std::vector<int64_t>{1, 0},
                                    Op::Settings(graph, tid + "_Transpose"));
  auto transpose_up =
      outplace_up->getInplaceVariant(Onnx::CustomOperators::TransposeInplace);

  auto transpose = transpose_up.get();
  transferBaseProperties(op, transpose);
  graph.moveIntoGraph(std::move(transpose_up));

  transpose->connectInTensor(TransposeOp::getInIndex(), tid);
  TensorId outId = tid + "/transposed";
  transpose->createAndConnectOutTensor(TransposeOp::getOutIndex(), outId);

  transpose->setup();
  return outId;
}

namespace {
PatternCreator<TiedGatherAccumulatePattern>
    tiedGatherAccumulater("TiedGatherAccumulate",
                          false,  // Off by default
                          false); // Not mandatory
} // namespace

/* Helpers */
namespace {

/**
 * Check either there is no matmul serialisation, or the mode is OutputChannels.
 */
bool canSerialiseTiedGather(const MatMulOp *matmul) {
  const auto &settings = matmul->getSerialiseSettings();
  return settings.factor <= 1 ||
         settings.mode == SerialiseSettings::Mode::None ||
         settings.mode == SerialiseSettings::Mode::OutputChannels;
}

/**
 * There is a transpose producer on exactly one of the ops.
 */
bool exactlyOneProducedByTranspose(const MatMulOp *matmul,
                                   const GatherOp *gather) {
  return tgutil::isProducedByTranspose(
             gather->input->tensor(GatherOp::dataInIndex())) !=
         tgutil::isProducedByTranspose(
             tgutil::maybeTraverseProducer<ReshapeBaseOp>(
                 ReshapeBaseOp::getInIndex(),
                 matmul->input->tensor(MatMulOp::getRhsInIndex())));
}

} // namespace

} // namespace popart
