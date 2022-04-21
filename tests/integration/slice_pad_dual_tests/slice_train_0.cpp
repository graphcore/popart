// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE slice_train_0

#include <algorithm>
#include <boost/test/unit_test.hpp>
#include <map>
#include <random>
#include <vector>
#include <popart/builder.hpp>
#include <popart/error.hpp>
#include <popart/ndarraywrapper.hpp>
#include <popart/session.hpp>
#include <popart/sgd.hpp>
#include <popart/testdevice.hpp>

BOOST_AUTO_TEST_CASE(SliceTrain0) {

  // model to train:
  //
  // weights
  //   |
  //   |---- slice ---=| == add ( weight slice 0, data) --------- l1
  //   |               |                                             \.
  //   |---- slice ---=| == add (weight slice 1, data)   ------- l1 -- final
  //                   |                                              /.
  //                   | == add (weight slice 1, weight slice 2)  -- l1
  // data ------------=|
  //
  //
  // data has shape   10 x 4.
  // weight has shape 25 x 4 if offset is false, and
  //                  25 x 5 if offset is true.
  //
  // The slices taken of weights have the data shape, i.e. 10 x 4.
  //
  // Weights when offset = true are,  ( 0 and 1 are the  2 sliced regions)
  //
  // . . . .
  // . . . .
  // . . . .
  // . . . .
  // . . . .
  // 0 0 0 0
  // 0 0 0 0
  // 0 0 0 0
  // 0 0 0 0
  //   |||
  // 1 1 1 1
  // 1 1 1 1
  // 1 1 1 1
  // 1 1 1 1
  //
  //
  // and when offset = false:
  //
  // . . . . .
  // . . . . .
  // . . . . .
  // . . . . .
  // . . . . .
  // . 0 0 0 0
  // . 0 0 0 0
  //   |||
  //   0 0 0 0
  //   0 0 0 0
  // 1 1 1 1 .
  // 1 1 1 1 .
  //   |||
  // 1 1 1 1 .
  // 1 1 1 1 .
  //
  // When offset is false, we expect the PadSum transform to be inapplicable.
  //

  /**
   * \param inplace Is the Inplacing Pattern enabled?
   *
   * \param padsum Is the PadSum Pattern, which converts sums of Pad outputs
   *               into concats where possible, enabled?
   *
   * \param offset See diagram above.
   * */
  auto getFinalWeights = [](bool inplace, bool padsum, bool offset) {
    using namespace popart;

    std::cout << "\n\n\nwith inplace = " << inplace << " padsum = " << padsum
              << " offset = " << offset << std::endl;

    std::map<std::string, std::pair<float, bool>> optParams;
    optParams.insert({"defaultLearningRate", {1.0f, true}});
    auto opt0 = SGD(optParams);

    // names of  weights used in model
    std::string w0name = "__w0__";

    auto builder = Builder::create();
    auto aiOnnx  = builder->aiOnnxOpset9();

    // sample shape: [sampleDim0, sampleDim1]
    int64_t sampleDim0 = 10;
    int64_t sampleDim1 = 4;
    std::vector<int64_t> sampleShape{sampleDim0, sampleDim1};
    TensorInfo sampleInfo{"FLOAT", sampleShape};

    // weigh shape:  [weightDim0, weightDim1]
    int64_t weightDim0 = 2 * sampleDim0 + 5;
    int64_t weightDim1 = sampleDim1 + 1 * (offset == true);
    auto nWeightVals   = weightDim0 * weightDim1;
    std::vector<int64_t> weightShape{weightDim0, weightDim1};
    TensorInfo weightInfo("FLOAT", weightShape);

    // a step of samples: [stepSize, batchSize, sampleDim0, sampleDim1]
    int64_t batchSize = 3;
    int64_t stepSize  = 1;

    std::vector<int64_t> batchShape{batchSize, sampleDim0, sampleDim1};
    TensorInfo batchInfo{"FLOAT", batchShape};

    std::vector<int64_t> stepDataShape{
        stepSize, batchSize, sampleDim0, sampleDim1};
    TensorInfo stepDataInfo{"FLOAT", stepDataShape};

    // Insert input Tensor into ONNX model
    auto input0 = builder->addInputTensor(batchInfo, "0tupni");

    // Note on randomness : generators are guaranteed to be platform
    // independent, so this will generate the same random values everywhere.
    // [0,1,2,3,4,5] - 2.5
    std::mt19937 gen(1011);
    auto getVal = [&gen]() { return static_cast<float>(gen() % 6) - 2.5; };

    // the initial weight values
    std::vector<float> weight0(nWeightVals);

    for (auto &x : weight0) {
      x = getVal();
    }

    // for reading back buffer for weight0
    std::vector<float> rb0(nWeightVals, -777.0f);

    // insert initialized weight Tensor into the ONNX model
    ConstVoidData cvd0({weight0.data(), weightInfo});
    auto w0Id = builder->addInitializedInputTensor(cvd0, w0name);
    WeightsIO weightsRead;
    weightsRead.insert(w0Id, {rb0.data(), weightInfo});

    // slice the weights into 2 parts, each the shape of a sample:
    auto slice0 = aiOnnx.slice(
        {w0Id}, {2 * sampleDim0, sampleDim1 + offset}, {sampleDim0, offset});
    auto slice1 = aiOnnx.slice({w0Id}, {sampleDim0, sampleDim1}, {0, 0});

    float lamb0 = 6.0;
    float lamb1 = 12.0;
    float lamb2 = 24.0;

    // input + weight slices, and sum of weight slices.
    // all with an l1 loss, them summed again:
    auto add0 = aiOnnx.add({slice0, input0});

    auto add1 = aiOnnx.add({slice1, add0});
    auto l1   = builder->aiGraphcoreOpset1().l1loss({add1}, lamb1);

    auto l0   = builder->aiGraphcoreOpset1().l1loss({add0}, lamb0);
    auto add2 = aiOnnx.add({slice0, slice1});
    auto l2   = builder->aiGraphcoreOpset1().l1loss({add2}, lamb2);

    auto lSum = aiOnnx.sum({l0, l1, l2});
    builder->addOutputTensor(lSum);

    auto proto    = builder->getModelProto();
    auto dataFlow = DataFlow(stepSize);

    SessionOptions userOptions;

    auto device = createTestDevice(TEST_TARGET, 1);

    auto session = popart::TrainingSession::createFromOnnxModel(
        proto,
        dataFlow,
        lSum,
        opt0,
        device,
        InputShapeInfo(),
        SessionOptions(),
        popart::Patterns(PatternsLevel::Default)
            .enableInPlace(inplace)
            .enablePattern("PadSum", true));

    session->prepareDevice();

    /*
    const auto sched = session->getIr().getOpSchedule({},
        RequireOptimalSchedule::Yes);
    for (auto x : sched) {
      std::cout << "\n   " << x->str();
    }
    std::cout << std::endl;
    */
    std::vector<float> v_input_x(stepDataInfo.nelms());
    for (auto &x : v_input_x) {
      x = getVal();
    }

    popart::NDArrayWrapper<float> input_x_wrapper(v_input_x.data(),
                                                  stepDataInfo);

    std::map<popart::TensorId, popart::IArray &> inputs = {
        {input0, input_x_wrapper}};

    popart::StepIO stepio(inputs, {});

    session->weightsFromHost();

    // run 1 with opt0
    session->run(stepio);
    session->run(stepio);

    // read final weights back
    session->weightsToHost();
    session->readWeights(weightsRead);

    return rb0;
  };

  //                                  inplace padsum offset
  const auto offset0 = getFinalWeights(true, true, true);
  const auto offset1 = getFinalWeights(true, false, true);
  const auto offset2 = getFinalWeights(false, false, true);
  const auto offset3 = getFinalWeights(false, true, true);

  if (offset0 != offset1 || offset0 != offset2 || offset0 != offset3) {
    throw popart::error("Offset cases do not all agree");
  }

  const auto nooffset0 = getFinalWeights(true, true, false);
  const auto nooffset1 = getFinalWeights(true, false, false);
  const auto nooffset2 = getFinalWeights(false, true, false);
  const auto nooffset3 = getFinalWeights(false, false, false);

  if (nooffset0 != nooffset1 || nooffset0 != nooffset2 ||
      nooffset0 != nooffset3) {
    throw popart::error("Offset cases do not all agree");
  }
}
