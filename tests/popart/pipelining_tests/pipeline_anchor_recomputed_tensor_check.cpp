// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE PipelineAnchorRecomputedTensor0

#include <algorithm>
#include <boost/test/unit_test.hpp>
#include <fstream>
#include <map>
#include <random>
#include <tuple>
#include <vector>

#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/devicemanager.hpp>
#include <popart/filereader.hpp>
#include <popart/inputshapeinfo.hpp>
#include <popart/ndarraywrapper.hpp>
#include <popart/op/ipucopy.hpp>
#include <popart/op/l1.hpp>
#include <popart/op/restore.hpp>
#include <popart/op/sigmoid.hpp>
#include <popart/op/stash.hpp>
#include <popart/optimizer.hpp>
#include <popart/session.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/tensornames.hpp>
#include <popart/testdevice.hpp>

BOOST_AUTO_TEST_CASE(PipelineAnchorRecomputedTensor0) {

  //  The test:
  //  Since, for pipelined models, the streaming of anchors to host occurs
  //  right at the end, we need to apply some transform in the Ir to ensure
  //  that the anchored tensor is not overwritten by recomputation.
  //  We check here that this works as expected.
  //
  //
  //  The model :
  //
  //         Input
  //           |
  //        sigmoid
  //           |
  //         scale
  //           |            IPU 0
  // - -  - -  - -  - -  - -  - -  - -  - -  - -  - -
  //           |            IPU 1
  //          L1 loss
  //
  //
  //  Sigmoid grad uses the output of sigmoid.
  //
  //  Without recomputation, this tensor is stashed.
  //
  //  With pipelining and recomputation, the Input is stashed, and for the
  //  Sigmoid grad in the backwards pass, the Input is restored, and the
  //  Sigmoid activation is recomputed

  enum class RunType {
    SingleDevice = 0,     // exact SGD on a single device
    ContinuousRecompPipe, // Continuous pipelining with recomputation
  };

  using namespace popart;

  auto getRecomputedResult = [](RunType rt) {
    int batchesPerStep     = 3;
    int64_t microBatchSize = 2;
    int64_t sampleSize     = 8;
    std::vector<int64_t> sampleShape{sampleSize};
    std::vector<int64_t> microBatchShape{microBatchSize, sampleSize};
    std::vector<int64_t> stepDataShape{
        batchesPerStep, microBatchSize, sampleSize};
    TensorInfo sampleInfo{"FLOAT", sampleShape};
    TensorInfo microBatchInfo{"FLOAT", microBatchShape};
    int64_t microBatchElms = sampleSize * microBatchSize;

    auto builder     = Builder::create();
    auto aiOnnx      = builder->aiOnnxOpset9();
    auto aiGraphcore = builder->aiGraphcoreOpset1();
    auto input0      = builder->addInputTensor(microBatchInfo);

    auto sig          = aiOnnx.sigmoid({input0});
    auto scale        = aiGraphcore.scale({sig}, 2.0f);
    TensorId actFinal = scale;
    auto loss         = L1Loss(actFinal, "l1LossVal", 0.1, ReductionType::Sum);

    int nIPUs = (rt != RunType::SingleDevice ? 2 : 1);
    std::map<std::string, std::string> deviceOpts{
        {"numIPUs", std::to_string(nIPUs)}};

    SessionOptions userOptions;
    if (rt == RunType::ContinuousRecompPipe) {
      userOptions.virtualGraphMode  = VirtualGraphMode::Manual;
      userOptions.enablePipelining  = true;
      userOptions.autoRecomputation = RecomputationType::Pipeline;

      builder->virtualGraph(sig, 0);
      builder->virtualGraph(scale, 0);
      loss.virtualGraph(nIPUs - 1);
    }
    auto art      = AnchorReturnType("All");
    auto inGradId = reservedGradientPrefix() + input0;
    auto dataFlow = DataFlow(batchesPerStep, {{sig, art}, {inGradId, art}});

    auto session = popart::TrainingSession::createFromOnnxModel(
        builder->getModelProto(),
        dataFlow,
        {&loss},
        ConstSGD(0.01),
        createTestDevice(TEST_TARGET, nIPUs),
        InputShapeInfo(),
        userOptions,
        popart::Patterns(PatternsLevel::Default));

    if (rt == RunType::ContinuousRecompPipe) {
      // Check our assumption, that the Sigmoid op is
      // annoted attribute "recompute: YES"
      auto opSchedule = session->getIr().getOpSchedule({});
      for (auto op : opSchedule) {
        if (dynamic_cast<SigmoidOp *>(op)) {
          BOOST_CHECK(op->settings.recomputeType == RecomputeType::Recompute);
        }
      }
    }

    session->prepareDevice();

    int64_t stepDataElms = microBatchElms * batchesPerStep;
    std::vector<float> sigData(stepDataElms);
    std::vector<float> inGradData(stepDataElms);
    popart::NDArrayWrapper<float> sigWrapper(sigData.data(), stepDataShape);
    popart::NDArrayWrapper<float> inGradWrapper(inGradData.data(),
                                                stepDataShape);
    std::map<popart::TensorId, popart::IArray &> anchors = {
        {sig, sigWrapper}, {inGradId, inGradWrapper}};

    // generate new samples
    std::default_random_engine eng(101);
    std::uniform_real_distribution<float> fdis(-1, 1);
    int64_t samplesPerStep = batchesPerStep * microBatchSize;
    std::vector<float> v_input_x(stepDataElms);
    for (int i = 0; i < samplesPerStep; ++i) {
      for (int j = 0; j < sampleSize; ++j) {
        auto stepIndex       = i * sampleSize + j;
        v_input_x[stepIndex] = fdis(eng);
      }
    }
    popart::NDArrayWrapper<float> input_x_wrapper(v_input_x.data(),
                                                  {"FLOAT", stepDataShape});
    std::map<popart::TensorId, popart::IArray &> inputs = {
        {input0, input_x_wrapper}};
    popart::StepIO stepio(inputs, anchors);

    // process the samples
    session->run(stepio);

    return sigData;
  };

  BOOST_CHECK(getRecomputedResult(RunType::SingleDevice) ==
              getRecomputedResult(RunType::ContinuousRecompPipe));
}
