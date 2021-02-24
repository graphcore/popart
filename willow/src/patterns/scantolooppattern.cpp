// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <memory>
#include <utility>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/dynamic/dynamicslice.hpp>
#include <popart/op/dynamic/dynamicupdate.hpp>
#include <popart/op/init.hpp>
#include <popart/op/loop.hpp>
#include <popart/op/reshape.hpp>
#include <popart/op/scan.hpp>
#include <popart/op/subtract.hpp>
#include <popart/patterns/scantolooppattern.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorindex.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/topocons.hpp>

namespace popart {

namespace {

TensorId getLoopIteratorId() { return reservedLoopIteratorPrefix(); }

TensorId getReversedLoopIteratorId() {
  std::stringstream ss;
  ss << reservedLoopIteratorPrefix();
  ss << "reversed";
  return ss.str();
}

TensorId getMaxTripCountM1Id() {
  std::stringstream ss;
  ss << reservedConstValuePrefix();
  ss << "maxTripCount-1";
  return ss.str();
}

} // namespace

bool ScanToLoopPattern::matches(Op *op) const {
  return op->isConvertibleTo<ScanOp>();
}

std::vector<const Tensor *> ScanToLoopPattern::touches(Op *) const {
  return {};
}

bool ScanToLoopPattern::apply(Op *op) const {

  if (ScanOp *scanOp = dynamic_cast<ScanOp *>(op)) {
    logging::pattern::debug("[ScanToLoopPattern] Converting {} to LoopOp",
                            scanOp->debugName());

    auto &graph   = scanOp->getGraph();
    auto &ir      = graph.getIr();
    auto settings = scanOp->settings;

    auto &scanSubgraph = scanOp->getCalledGraph();

    auto loopSubgraphId    = ir.createUniqueSubgraphId({"loop"});
    auto &loopSubgraph     = ir.createGraph(loopSubgraphId);
    auto loopSubgraphScope = loopSubgraph.getScope();

    // Add mandatory loop iterator tensor to subgraph (is not an output)
    TensorId loopSgItId    = loopSubgraph.addScope(getLoopIteratorId());
    TensorId loopSgRevItId = loopSubgraph.addScope(getReversedLoopIteratorId());
    loopSubgraph.addInput(loopSgItId, TensorInfo(DataType::INT32, {}));

    // Add mandatory loop condition tensor to subgraph (is also an output)
    TensorId loopSgCondId = loopSubgraph.addScope(reservedLoopCondPrefix());
    loopSubgraph.addInput(loopSgCondId, TensorInfo(DataType::BOOL, {}));
    loopSubgraph.markAsOutput(loopSgCondId);

    // Add reverse loop iterator
    TensorId loopSgMaxTripCountM1Id =
        loopSubgraph.addScope(getMaxTripCountM1Id());
    TensorInfo indexTensorInfo(DataType::INT32, {1});
    std::vector<int32_t> idData(1, scanOp->getTripCountValue() - 1);
    loopSubgraph.getTensors().addConstInit(
        loopSgMaxTripCountM1Id,
        indexTensorInfo,
        reinterpret_cast<void *>(idData.data()));

    auto subtractSettings  = settings;
    subtractSettings.scope = loopSubgraphScope;
    auto subtractOpUp =
        std::make_unique<SubtractOp>(Onnx::Operators::Sub_7, subtractSettings);
    SubtractOp *subtractOp = subtractOpUp.get();
    loopSubgraph.moveIntoGraph(std::move(subtractOpUp));

    subtractOp->connectInTensor(SubtractOp::getArg0InIndex(),
                                loopSgMaxTripCountM1Id);
    subtractOp->connectInTensor(SubtractOp::getArg1InIndex(), loopSgItId);
    subtractOp->createAndConnectOutTensor(SubtractOp::getOutIndex(),
                                          loopSgRevItId);
    subtractOp->setup();

    auto loopOpUp = std::make_unique<LoopOp>(
        Onnx::Operators::Loop_11, settings, loopSubgraph);
    LoopOp *loopOp = loopOpUp.get();
    graph.moveIntoGraph(std::move(loopOpUp));

    loopOp->setTripCountValue(scanOp->getTripCountValue());

    int N = scanOp->getNumVariables();
    int K = scanOp->getNumScanOutputs();
    int M = scanOp->getNumScanInputs();
    int L = scanOp->getNumImplicitInputs();

    std::map<TensorId, TensorId> tensorRemap;

    // Add variable input tensors -> explicit loop inputs
    for (int n = 0; n < N; ++n) {
      auto varInId        = scanOp->inId(n);
      TensorId scanSgInId = scanSubgraph.getInputId(n);
      TensorId loopSgInId =
          loopSubgraph.addScope(scanSubgraph.removeScope(scanSgInId));
      tensorRemap[scanSgInId] = loopSgInId;
      loopOp->addLoopInput(
          LoopOp::getFirstInputInIndex() + n, varInId, loopSgInId, true);
    }

    // Add scan inputs -> implicit loop inputs
    for (int m = 0; m < M; ++m) {
      auto scanInId       = scanOp->inId(N + m);
      TensorId scanSgInId = scanSubgraph.getInputId(N + m);
      TensorId loopSgInId =
          loopSubgraph.addScope(scanSubgraph.removeScope(scanSgInId));
      tensorRemap[scanSgInId] = loopSgInId;

      TensorId loopSgInIdTmp0 = ir.createIntermediateTensorId(loopSgInId);
      TensorId loopSgInIdTmp1 = ir.createIntermediateTensorId(loopSgInId);

      std::vector<int64_t> axesv(1, scanOp->getScanInputAxis(m));
      std::vector<int64_t> sizesv(1, 1);

      Tensor *scanSgIn = scanSubgraph.getTensors().get(scanSgInId);

      Op::Settings sliceSettings = settings;

      for (Op *c : scanSgIn->consumers.getOps()) {
        sliceSettings = c->settings;
        break;
      }
      sliceSettings.scope = loopSubgraphScope;

      // Slice the scan input
      std::unique_ptr<DynamicSliceOp> sliceOpUp =
          std::make_unique<DynamicSliceOp>(
              Onnx::CustomOperators::DynamicSlice_1,
              axesv,
              sizesv,
              true,
              sliceSettings);
      DynamicSliceOp *sliceOp = sliceOpUp.get();
      loopSubgraph.moveIntoGraph(std::move(sliceOpUp));

      // Drop axis dimension from sliced input if rank > 1
      // This squeezing behaviour may not be specified in the standard,
      // but some TF2ONNX models seem to depend on it
      std::unique_ptr<ReshapeOp> reshapeOpUp =
          std::make_unique<ReshapeOp>(Onnx::AiOnnx::OpSet11::Reshape,
                                      scanSgIn->info.shape(),
                                      sliceSettings);
      Op *reshapeOp = reshapeOpUp.get();
      loopSubgraph.moveIntoGraph(std::move(reshapeOpUp));

      loopOp->addLoopInput(LoopOp::getFirstInputInIndex() + N + m,
                           scanInId,
                           loopSgInIdTmp0,
                           true);

      sliceOp->connectInTensor(DynamicSliceOp::getInIndex(), loopSgInIdTmp0);
      sliceOp->connectInTensor(DynamicSliceOp::getIndexInIndex(),
                               scanOp->isScanInputReversed(m) ? loopSgRevItId
                                                              : loopSgItId);
      sliceOp->createAndConnectOutTensor(DynamicSliceOp::getOutIndex(),
                                         loopSgInIdTmp1);
      sliceOp->setup();

      reshapeOp->connectInTensor(ReshapeOp::getInIndex(), loopSgInIdTmp1);
      reshapeOp->createAndConnectOutTensor(ReshapeOp::getOutIndex(),
                                           loopSgInId);
      reshapeOp->setup();
    }

    // Add implicit scan inputs -> implicit loop inputs
    for (int l = 0; l < L; ++l) {
      auto implicitInId   = scanOp->inId(N + M + l);
      TensorId scanSgInId = scanSubgraph.getInputId(N + M + l);
      TensorId loopSgInId =
          loopSubgraph.addScope(scanSubgraph.removeScope(scanSgInId));
      tensorRemap[scanSgInId] = loopSgInId;
      loopOp->addLoopInput(LoopOp::getFirstInputInIndex() + N + M + l,
                           implicitInId,
                           loopSgInId,
                           true);
    }

    // Move over subgraph Ops
    for (Op *scanSgOp :
         scanSubgraph.getOpSchedule({}, RequireOptimalSchedule::No)) {
      auto loopSgOpUp = scanSgOp->clone();
      Op *loopSgOp    = loopSgOpUp.get();
      loopSubgraph.moveIntoGraph(std::move(loopSgOpUp));
      loopSgOp->setScope(loopSubgraphScope);

      // Resolve Op inputs
      for (auto &in : scanSgOp->input->tensorMap()) {
        loopSgOp->connectInTensor(in.first, tensorRemap.at(in.second->id));
      }

      // Resolve Op outputs
      for (auto &out : scanSgOp->output->tensorMap()) {
        TensorId loopSgOutId =
            loopSubgraph.addScope(scanSubgraph.removeScope(out.second->id));
        tensorRemap[out.second->id] = loopSgOutId;
        loopSgOp->createAndConnectOutTensor(out.first, loopSgOutId);
      }
      loopSgOp->setup();
    }

    // Process variable output tensors
    for (int n = 0; n < N; ++n) {
      auto varOutId        = scanOp->outId(n);
      TensorId scanSgOutId = scanSubgraph.getOutputId(n);
      scanOp->disconnectOutTensor(graph.getTensors().get(varOutId));
      loopOp->addLoopOutput(n, varOutId, tensorRemap.at(scanSgOutId), true);
    }

    // Create InitOps for the scan outputs, update scan outputs
    for (int k = 0; k < K; ++k) {
      auto scanOut   = scanOp->outTensor(N + k);
      auto scanOutId = scanOut->id;

      TensorId scanSgOutId     = scanSubgraph.getOutputId(N + k);
      Tensor *scanSgOut        = scanSubgraph.getTensors().get(scanSgOutId);
      TensorId loopSgOutId     = tensorRemap.at(scanSgOutId);
      TensorId loopSgUpdatedId = ir.createIntermediateTensorId(loopSgOutId);

      TensorId initId       = ir.createIntermediateTensorId(scanOutId);
      TensorId loopSgInitId = loopSubgraph.addScope(initId);

      // For reshaping
      TensorId loopSgInitRId = ir.createIntermediateTensorId(loopSgInitId);
      TensorId loopSgOutRId  = ir.createIntermediateTensorId(loopSgOutId);
      TensorId loopSgUpdatedRId =
          ir.createIntermediateTensorId(loopSgUpdatedId);

      // InitOp as a "producer" Op
      auto initOpUp  = std::make_unique<InitOp>(Onnx::CustomOperators::Init_1,
                                               scanOut->info,
                                               scanOut->tensorType(),
                                               InitType::NoInit,
                                               settings);
      InitOp *initOp = initOpUp.get();
      graph.moveIntoGraph(std::move(initOpUp));
      initOp->createAndConnectOutTensor(InitOp::getOutIndex(), initId);
      initOp->setup();

      logging::pattern::trace(
          "[ScanToLoopPattern] Adding {} for scan output {} (info {})",
          initOp->debugName(),
          scanOut->id,
          scanOut->info);

      loopOp->addLoopInput(
          LoopOp::getFirstInputInIndex() + N + k, initId, loopSgInitId, false);
      scanOp->disconnectOutTensor(scanOut);

      auto axis = scanOp->getScanOutputAxis(k);

      Op::Settings updateSettings = settings;

      if (scanSgOut->hasProducer()) {
        updateSettings = scanSgOut->getProducer()->settings;
      }
      updateSettings.scope = loopSubgraphScope;

      if (scanSgOut->info.shape().at(axis) > 1) {
        // Reshape the init tensor
        auto inShape = scanOut->info.shape();
        inShape[axis] /= scanOp->getTripCountValue();
        inShape.insert(inShape.begin() + axis, scanOp->getTripCountValue());
        logging::pattern::trace("[ScanToLoopPattern] Reshaping {} {} -> {} {}",
                                loopSgInitId,
                                scanOut->info.shape(),
                                loopSgInitRId,
                                inShape);
        std::unique_ptr<ReshapeOp> reshapeInOpUp = std::make_unique<ReshapeOp>(
            Onnx::AiOnnx::OpSet11::Reshape, inShape, updateSettings);
        Op *reshapeInOp = reshapeInOpUp.get();
        loopSubgraph.moveIntoGraph(std::move(reshapeInOpUp));

        reshapeInOp->connectInTensor(ReshapeOp::getInIndex(), loopSgInitId);
        reshapeInOp->createAndConnectOutTensor(ReshapeOp::getOutIndex(),
                                               loopSgInitRId);
        reshapeInOp->setup();
        loopSgInitId = loopSgInitRId;

        // Reshape the scan body output / slice tensor
        auto sliceShape = scanSgOut->info.shape();
        sliceShape.insert(sliceShape.begin() + axis, 1);
        logging::pattern::trace("[ScanToLoopPattern] Reshaping {} {} -> {} {}",
                                loopSgOutId,
                                scanSgOut->info.shape(),
                                loopSgOutRId,
                                sliceShape);
        std::unique_ptr<ReshapeOp> reshapeSliceOpUp =
            std::make_unique<ReshapeOp>(
                Onnx::AiOnnx::OpSet11::Reshape, sliceShape, updateSettings);
        Op *reshapeSliceOp = reshapeSliceOpUp.get();
        loopSubgraph.moveIntoGraph(std::move(reshapeSliceOpUp));

        reshapeSliceOp->connectInTensor(ReshapeOp::getInIndex(), loopSgOutId);
        reshapeSliceOp->createAndConnectOutTensor(ReshapeOp::getOutIndex(),
                                                  loopSgOutRId);
        reshapeSliceOp->setup();
        loopSgOutId = loopSgOutRId;
      }

      std::vector<int64_t> axesv(1, axis);
      std::vector<int64_t> sizesv(1, 1);

      // Update the scan output
      std::unique_ptr<DynamicUpdateOp> updateOpUp =
          std::make_unique<DynamicUpdateOp>(
              Onnx::CustomOperators::DynamicUpdate_1,
              axesv,
              sizesv,
              true,
              updateSettings);
      DynamicUpdateOp *updateOp = updateOpUp.get();
      loopSubgraph.moveIntoGraph(std::move(updateOpUp));

      updateOp->connectInTensor(DynamicUpdateOp::getInIndex(), loopSgOutId);
      updateOp->connectInTensor(DynamicUpdateOp::getIndexInIndex(),
                                scanOp->isScanOutputReversed(k) ? loopSgRevItId
                                                                : loopSgItId);
      updateOp->connectInTensor(DynamicUpdateOp::getUpdateInIndex(),
                                loopSgInitId);
      updateOp->createAndConnectOutTensor(DynamicUpdateOp::getOutIndex(),
                                          loopSgUpdatedId);
      updateOp->setup();

      if (scanSgOut->info.shape().at(axis) > 1) {
        // Reshape the LoopOp/ScanOp output
        std::unique_ptr<ReshapeOp> reshapeOutOpUp =
            std::make_unique<ReshapeOp>(Onnx::AiOnnx::OpSet11::Reshape,
                                        scanOut->info.shape(),
                                        updateSettings);
        Op *reshapeOutOp = reshapeOutOpUp.get();
        loopSubgraph.moveIntoGraph(std::move(reshapeOutOpUp));

        reshapeOutOp->connectInTensor(ReshapeOp::getInIndex(), loopSgUpdatedId);
        reshapeOutOp->createAndConnectOutTensor(ReshapeOp::getOutIndex(),
                                                loopSgUpdatedRId);
        reshapeOutOp->setup();
        loopSgUpdatedId = loopSgUpdatedRId;
      }

      loopOp->addLoopOutput(N + k, scanOutId, loopSgUpdatedId, false);
    }

    // Transfer topocons
    graph.topoCons->transfer(scanOp, loopOp, true);

    // Remove the ScanOp
    scanOp->disconnectAllInputs();
    scanOp->disconnectAllOutputs();
    graph.eraseOp(scanOp->id);

    if (scanSubgraph.getCallSiteOps().empty()) {
      ir.removeGraph(scanSubgraph.id);
    }

    loopOp->setup();

    for (auto &output : loopOp->output->tensorMap()) {
      logging::pattern::trace(
          "[ScanToLoopPattern] LoopOp output {}: {} (info {})",
          output.first,
          output.second->id,
          output.second->info);
    }

    logging::pattern::debug(
        "[ScanToLoopPattern] Converted ScanOp to {} "
        "(explicit inputs: {}, implicit inputs: {}, outputs: {})",
        loopOp->debugName(),
        loopOp->numExplicitInputs(),
        loopOp->numImplicitInputs(),
        loopOp->output->n());
    return true;
  } else {
    throw error("[ScanToLoopPattern] Op {} is not a ScanOp.", op->debugName());
  }
  return false;
}

namespace {
static PatternCreator<ScanToLoopPattern>
    ScanToLoopPattern(PreAliasPatternType::ScanToLoop,
                      "ScanToLoop",
                      /* enabled = */ true,
                      /* mandatory = */ true);

}

} // namespace popart
