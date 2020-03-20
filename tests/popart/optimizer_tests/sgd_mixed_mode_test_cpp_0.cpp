// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE sgd_mixed_mode_test_0

#include <algorithm>
#include <boost/test/unit_test.hpp>
#include <map>
#include <random>
#include <tuple>
#include <vector>

#define protected public
#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/devicemanager.hpp>
#include <popart/filereader.hpp>
#include <popart/inputshapeinfo.hpp>
#include <popart/ndarraywrapper.hpp>
#include <popart/op/ipucopy.hpp>
#include <popart/op/l1.hpp>
#include <popart/op/restore.hpp>
#include <popart/op/sgd0varupdate.hpp>
#include <popart/op/stash.hpp>
#include <popart/optimizer.hpp>
#include <popart/session.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/tensornames.hpp>
#include <popart/testdevice.hpp>
#undef protected

BOOST_AUTO_TEST_CASE(SgdMixedModeTest0) {

  using namespace popart;

  // names of 3 weights used in model
  std::string w0name = "__w0__";
  std::string w1name = "__w1__";
  std::string w2name = "__w2__";

  class Expectation {
  public:
    // after one run with SGD opt0, and one run with SGD opt1, what do we expect
    // the final value of the weight to be?
    float finalValue;

    // Given opt0 (and opt1) do we expect the computation to be done with
    // constant optimizer values?
    bool scaledLearningRateIsConst;
    bool weightDecayScaleFactorIsConst;
  };

  auto test = [w0name, w1name, w2name](
                  const SGD &opt0,       // initial Optimizer
                  const SGD &opt1,       // Optimizer to switch to
                  const Expectation &e0, // expectation for weight 0
                  const Expectation &e1, // expectation for weight 1
                  const Expectation &e2  // expectation for weight 2
              ) {
    // Model
    // ----
    //
    // loss = l1_loss((((input + w0) + w1) + w2))
    //
    // where input is small positive and w0, w1, w2 are large positive
    //

    auto builder = Builder::create();
    auto aiOnnx  = builder->aiOnnxOpset9();

    int64_t sampleDim = 1;
    int64_t batchSize = 1;
    int64_t stepSize  = 1;

    std::vector<int64_t> batchShape{batchSize, sampleDim};
    TensorInfo batchInfo{"FLOAT", batchShape};

    std::vector<int64_t> sampleShape{sampleDim};
    TensorInfo sampleInfo{"FLOAT", sampleShape};

    std::vector<int64_t> stepDataShape{stepSize, batchSize, sampleDim};
    TensorInfo stepDataInfo{"FLOAT", stepDataShape};

    auto input0 = builder->addInputTensor(batchInfo, "0tupni");

    WeightsIO weightsRead;

    std::vector<float> weight0(sampleDim, 100.0f);
    // read-back buffer for weight0
    std::vector<float> rb0(sampleDim, -777.0f);
    ConstVoidData cvd0({weight0.data(), sampleInfo});

    auto w0Id = builder->addInitializedInputTensor(cvd0, w0name);
    weightsRead.insert(w0Id, {rb0.data(), sampleInfo});

    std::vector<float> weight1(sampleDim, 200.0f);
    std::vector<float> rb1(sampleDim, -777.0f);
    ConstVoidData cvd1({weight1.data(), sampleInfo});
    auto w1Id = builder->addInitializedInputTensor(cvd1, w1name);
    weightsRead.insert(w1Id, {rb1.data(), sampleInfo});

    std::vector<float> weight2(sampleDim, 300.0f);
    std::vector<float> rb2(sampleDim, -777.0f);
    ConstVoidData cvd2({weight2.data(), sampleInfo});
    auto w2Id = builder->addInitializedInputTensor(cvd2, w2name);
    weightsRead.insert(w2Id, {rb2.data(), sampleInfo});

    auto add0 = aiOnnx.add({w0Id, input0});
    auto add1 = aiOnnx.add({w1Id, add0});
    auto add2 = aiOnnx.add({w2Id, add1});

    builder->addOutputTensor(add2);
    auto proto    = builder->getModelProto();
    auto dataFlow = DataFlow(stepSize);

    float lambda = 1.0;
    auto loss    = std::unique_ptr<Loss>(
        new L1Loss(add2, "l1LossVal", lambda, ReductionType::SUM));

    SessionOptions userOptions;

    auto device = createTestDevice(TEST_TARGET, 1);

    auto session = popart::TrainingSession::createFromOnnxModel(
        proto,
        dataFlow,
        {loss.get()},
        opt0, // construct with opt0, will switch to opt1 later
        device,
        InputShapeInfo(),
        SessionOptions(),
        popart::Patterns(PatternsLevel::DEFAULT));

    session->prepareDevice();
    std::vector<float> v_input_x(stepDataInfo.nelms(), 3.1415);

    popart::NDArrayWrapper<float> input_x_wrapper(v_input_x.data(),
                                                  stepDataInfo);

    std::map<popart::TensorId, popart::IArray &> inputs = {
        {input0, input_x_wrapper}};

    popart::StepIO stepio(inputs, {});

    session->weightsFromHost();

    // run 1 with opt0
    session->optimizerFromHost();
    session->run(stepio);

    // run 2 with opt1
    session->updateOptimizer(&opt1);
    session->optimizerFromHost();
    session->run(stepio);

    // read final weights back
    session->weightsToHost();
    session->readWeights(weightsRead);

    std::cout << "Values read-back from device for weights 0,1,2:" << std::endl;
    std::cout << rb0[0] << "  " << rb1[0] << "  " << rb2[0] << "." << std::endl;

    // All the SGD0VarUpdateOps, in no particular order
    std::vector<SGD0VarUpdateOp *> sgdOpsOOO;
    for (auto op : session->ir.getOpSchedule({})) {
      auto asSgd = dynamic_cast<SGD0VarUpdateOp *>(op);
      if (asSgd) {
        sgdOpsOOO.push_back(asSgd);
      }
    }
    BOOST_CHECK(sgdOpsOOO.size() == 3);

    std::vector<SGD0VarUpdateOp *> sgdOps = sgdOpsOOO;
    for (auto op : sgdOpsOOO) {
      auto id = op->inId(op->getVarToUpdateInIndex());
      if (id == w0name) {
        sgdOps[0] = op;
      } else if (id == w1name) {
        sgdOps[1] = op;
      } else if (id == w2name) {
        sgdOps[2] = op;
      } else {
        throw error("failed to determine input id in sgd test");
      }
    }

    // check that expected values are the same as read-back values
    BOOST_CHECK(e0.finalValue == rb0[0]);
    BOOST_CHECK(e1.finalValue == rb1[0]);
    BOOST_CHECK(e2.finalValue == rb2[0]);

    BOOST_CHECK(e0.scaledLearningRateIsConst == sgdOps[0]->initSlr0.isConst());
    BOOST_CHECK(e1.scaledLearningRateIsConst == sgdOps[1]->initSlr0.isConst());
    BOOST_CHECK(e2.scaledLearningRateIsConst == sgdOps[2]->initSlr0.isConst());

    BOOST_CHECK(e0.weightDecayScaleFactorIsConst ==
                sgdOps[0]->initWdsf0.isConst());
    BOOST_CHECK(e1.weightDecayScaleFactorIsConst ==
                sgdOps[1]->initWdsf0.isConst());
    BOOST_CHECK(e2.weightDecayScaleFactorIsConst ==
                sgdOps[2]->initWdsf0.isConst());
  };

  // Test case 1
  // ------------
  std::cout << "TEST CASE 1" << std::endl;
  OptimizerValue defaultWeightDecay{0, false};
  OptimizerValue defaultLearningRate{0.1, true};
  OptimizerValue lossScaling{10, true};

  std::map<std::string, std::pair<float, bool>> optParams;
  optParams.insert({"defaultWeightDecay", {0.0f, false}});
  optParams.insert({"defaultLearningRate", {0.1f, true}});
  optParams.insert({"lossScaling", {10.0f, true}});

  auto opt0 = SGD(optParams);
  opt0.insertSpecific(
      w1name, {{"learningRate", {0.2, false}}, {"weightDecay", {0.0f, true}}});

  // same as opt1, but increased learning rate for weight 1
  auto opt1 = SGD(optParams);

  opt1.insertSpecific(
      w1name, {{"learningRate", {0.5, false}}, {"weightDecay", {0, true}}});

  // weight 0 uses the global optimizer values
  Expectation e0({
      100 - 0.1 - 0.1, // initial value - (lr run 1) - (lr run 2)
      true,            // constant scaled learning rate
      false            // non-constant weight decay scale factor
  });

  // weight 1 uses local optimizer values
  Expectation e1({
      200 - 0.2 - 0.5, // initial - (lr 1) - (lr 2)
      false,           // non-constant scaled learning rate
      false // non-constant scaled weight decay - this is because of the
            // dependance on the learning rate, which is non-constant
  });

  Expectation e2({300 - 0.1 - 0.1, true, false});

  test(opt0, opt1, e0, e1, e2);

  // Test case 2
  // -----------
  // We confirm that a non-const lossScaling results in all learning rates being
  // non-const
  std::cout << "TEST CASE 2" << std::endl;

  opt0 = SGD({{"defaultLearningRate", {2, true}},
              {"defaultWeightDecay", {0, true}},
              {"lossScaling", {20, false}}});

  opt0.insertSpecific(
      w1name, {{"learningRate", {4.0, true}}, {"weightDecay", {0.125, true}}});

  opt1 = opt0;

  // weight 0 uses the global optimizer values
  e0 = {
      100 - 2.0 - 2.0,
      false, // scaled learning rate is not constant, becase loss scaling isn't
      true   // weight decay scaling factor is constant
  };

  // 0.125 * 4.0 = 0.5 (half-life of 1)
  // 200 * (1 - 0.5) - 4 = 96
  // 96  * (1 - 0.5) - 4 = 44
  //
  // weight 1 uses local optimizer values
  e1 = {44, false, true};

  e2 = {300 - 2.0 - 2.0, false, true};

  test(opt0, opt1, e0, e1, e2);
}
