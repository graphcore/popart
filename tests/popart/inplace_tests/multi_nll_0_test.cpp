// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE MultiNll0InplaceTest

#include <../random_util.hpp>
#include <boost/filesystem.hpp>
#include <boost/test/unit_test.hpp>
#include <chrono>
#include <complex>
#include <iostream>
#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/devicemanager.hpp>
#include <popart/filereader.hpp>
#include <popart/inputshapeinfo.hpp>
#include <popart/ir.hpp>
#include <popart/names.hpp>
#include <popart/ndarraywrapper.hpp>
#include <popart/op/add.hpp>
#include <popart/op/identity.hpp>
#include <popart/op/nll.hpp>
#include <popart/optimizer.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/stepio.hpp>
#include <popart/tensordata.hpp>
#include <popart/tensornames.hpp>
#include <popart/testdevice.hpp>

#define protected public
#include <popart/session.hpp>
#undef protected

using namespace popart;

BOOST_AUTO_TEST_CASE(test) {

  // (see T11824, T12152, D16243)
  //
  //
  // Input ---|          |--slice--softmax--nll
  //          |--matmul--|
  // Weight --|          |--slice--softmax--nll
  //
  //  backwards :
  //
  //   ... sliceGrad---|
  //                   |--Add-- (the gradient of the output of matmul).
  //   ... sliceGrad---|
  //
  // We disable PadSum pattern to retain the slice grads.

  auto getFinalWeights = [](int batchesPerStep,
                            const std::array<float, 6 * 2>
                                &vWeight, // initial weights
                            bool doInplace) {
    auto builder     = Builder::create();
    auto aiOnnx      = builder->aiOnnxOpset9();
    auto aiGraphcore = builder->aiGraphcoreOpset1();

    int batchSize = 1;

    // data
    TensorInfo dataSampleInfo{"FLOAT", std::vector<int64_t>{4, 6}};
    TensorInfo dataBatchInfo{"FLOAT", std::vector<int64_t>{batchSize, 4, 6}};

    TensorInfo dataStepInfo{
        "FLOAT", std::vector<int64_t>{batchesPerStep, batchSize, 4, 6}};
    auto input = builder->addInputTensor(dataBatchInfo);

    // weights
    TensorInfo weightInfo{"FLOAT", std::vector<int64_t>{6, 2}};
    ConstVoidData cvWeight = {vWeight.data(), weightInfo};
    auto weight            = builder->addInitializedInputTensor(cvWeight);

    // labels 0 and 1
    TensorInfo labelSampleInfo{"INT32", std::vector<int64_t>{1}};
    TensorInfo labelBatchInfo{"INT32", std::vector<int64_t>{batchSize, 1}};
    TensorInfo labelStepInfo{
        "INT32", std::vector<int64_t>{batchesPerStep, batchSize, 1}};

    // matmul (1, 4, 2) (why not (4, 2)?)
    auto mmOut = aiOnnx.matmul({input, weight});

    auto slice0 =
        aiOnnx.slice({mmOut}, {1, 4, 1}, {0, 0, 0}, {0, 1, 2}, "slc0");
    auto reshape0 = builder->reshape_const(aiOnnx, {slice0}, {1, 4}, "rsh0");
    auto sm0      = aiOnnx.softmax({reshape0}, 1, "sm0");
    auto label0   = builder->addInputTensor(labelSampleInfo);
    auto nll0     = aiGraphcore.nllloss({sm0, label0}, ReductionType::Mean);

    auto slice1 =
        aiOnnx.slice({mmOut}, {1, 4, 2}, {0, 0, 1}, {0, 1, 2}, "slc1");
    auto reshape1 = builder->reshape_const(aiOnnx, {slice1}, {1, 4}, "rsh1");
    auto sm1      = aiOnnx.softmax({reshape1}, 1, "sm1");
    auto label1   = builder->addInputTensor(labelSampleInfo);
    auto nll1     = aiGraphcore.nllloss({sm1, label1}, ReductionType::Mean);

    auto finalLoss = aiOnnx.sum({nll0, nll1});

    auto device = createTestDevice(TEST_TARGET, 1, 20);

    auto session = popart::TrainingSession::createFromOnnxModel(
        builder->getModelProto(),
        DataFlow(batchesPerStep),
        finalLoss,
        ConstSGD(1.0),
        device,
        InputShapeInfo(),
        SessionOptions(),
        popart::Patterns(PatternsLevel::Default)
            .enableInPlace(doInplace)
            .enablePattern("PadSum", false));

    auto sched = session->ir.getOpSchedule({});
    std::cout << "The op schedule with inplace=" << doInplace << " is :\n";
    int nAdds = 0;
    for (const auto *op : sched) {
      std::cout << op->str() << std::endl;
      if (dynamic_cast<const AddOp *>(op) ||
          dynamic_cast<const AddLhsInplaceOp *>(op) ||
          dynamic_cast<const AddRhsInplaceOp *>(op)) {
        ++nAdds;
      }
    }

    // We expect a single add in the backwards pass, which should be inplaced
    // IFF inplacing is enabled. Note that in the future we might have an
    // optimization which removes this Add, in which case this check will fail
    // (T12632)
    BOOST_CHECK(nAdds == 1);

    int expectedAddInplace = doInplace ? 1 : 0;
    BOOST_CHECK(
        session->ir.opsOfType(Onnx::CustomOperators::AddLhsInplace).size() +
            session->ir.opsOfType(Onnx::CustomOperators::AddRhsInplace)
                .size() ==
        expectedAddInplace);

    session->prepareDevice();

    auto seed = 1011;
    DefaultRandomEngine eng(seed);
    UniformRealDistribution<float> fdis(0.f, 0.5f);
    UniformIntDistribution<int> idis(0, 3);

    WeightsIO weightsRead;
    std::vector<float> readBackWeights(weightInfo.nelms(), -777.0f);
    weightsRead.insert(weight, {readBackWeights.data(), weightInfo});

    std::vector<float> vInput(dataStepInfo.nelms(), 0.1);
    popart::NDArrayWrapper<float> inputWrapper(vInput.data(), dataStepInfo);
    for (auto &v : vInput) {
      v = fdis(eng);
    }

    std::vector<int> vLabel0(labelStepInfo.nelms(), 2);
    popart::NDArrayWrapper<int> label0Wrapper(vLabel0.data(), labelStepInfo);
    for (auto &v : vLabel0) {
      v = idis(eng);
    }

    std::vector<int> vLabel1(labelStepInfo.nelms(), 3);
    popart::NDArrayWrapper<int> label1Wrapper(vLabel1.data(), labelStepInfo);
    for (auto &v : vLabel1) {
      v = idis(eng);
    }

    std::map<popart::TensorId, popart::IArray &> inputs = {
        {input, inputWrapper},
        {label0, label0Wrapper},
        {label1, label1Wrapper}};

    popart::StepIO stepio(inputs, {});

    session->weightsFromHost();

    session->run(stepio);
    session->weightsToHost();
    session->readWeights(weightsRead);

    return readBackWeights;
  };

  // generate random input data
  auto seed = 1011;
  DefaultRandomEngine eng(seed);
  UniformRealDistribution<float> fdis(0.f, 0.5f);
  std::array<float, 6 * 2> vWeight;
  for (auto &val : vWeight) {
    val = fdis(eng);
  }

  auto weights0 = getFinalWeights(2, vWeight, false);
  auto weights1 = getFinalWeights(2, vWeight, true);

  float absErr = 0.0f;
  for (int i = 0; i < weights1.size(); ++i) {
    absErr += std::abs(weights0[i] - weights1[i]);
  }
  std::cout << "Absolute error is " << absErr << std::endl;
  BOOST_CHECK(absErr < 1e-6);
}
