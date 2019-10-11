#define BOOST_TEST_MODULE sgd_mixed_mode_test_1

#include <algorithm>
#include <array>
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
#undef protected

namespace {
// names of 2 weights used in model
constexpr const char *w0name = "__w0__";
constexpr const char *w1name = "__w1__";
} // namespace

float getAbsDiff(float expected, float observed) {
  float absv = std::abs(expected - observed);
  std::cout << "Expected=" << expected << ", observed=" << observed
            << " with absolute difference=" << absv << std::endl;
  return absv;
}

// The pytorch update equations:
//
// (1) g = g + wd * w
// (2) v = v * mm + (1 - dp) * g
// (3) w = w - lr * v
//
void pytorchUpdate(float &w,
                   float &g,
                   float &v,
                   float wd,
                   float mm,
                   float dp,
                   float lr) {

  g = +1.0f; // from the model below
  g += wd * w;
  v *= mm;
  v += (1.0f - dp) * g;
  w -= lr * v;
}

void laggedPytorchUpdate(float &w,
                         float &g,
                         float &v,
                         float wd,
                         float mm,
                         float dp,
                         float lr) {

  g = +1.0f; // from the model below
  v = v + (1.0f - dp) * g;
  w = w - lr * v;
  v = v * mm + (1.0f - dp) * wd * w;
}

void laggedPytorchUpdateWithScaling(float &w,
                                    float &g,
                                    float &v,
                                    float wd,
                                    float mm,
                                    float dp,
                                    float lr,
                                    float vs,
                                    float ls) {

  g = +1.0f * ls;
  v = v + vs * (1.0f - dp) * g / ls;
  w = w - lr * v / vs;
  v = v * mm + vs * (1.0f - dp) * wd * w;
}

std::array<float, 2>
getResults(const popart::SGD &opt0, // initial Optimizer
           const popart::SGD &opt1, // Optimizer to switch to
           const popart::SGD &opt2, // Last Optimizer
           bool gradAccl = false) {

  using namespace popart;
  // Model
  // ----
  //
  // loss = l1_loss((input + w0) + w1)
  //
  // where input is small positive and w0, w1 are large positive
  //

  auto builder = Builder::create();
  auto aiOnnx  = builder->aiOnnxOpset9();

  int64_t sampleDim      = 1;
  int64_t microBatchSize = 1;
  int64_t batchSize      = (gradAccl ? 2 : 1) * microBatchSize;
  int64_t stepSize       = 1;

  std::vector<int64_t> microBatchShape{microBatchSize, sampleDim};
  TensorInfo microBatchInfo{"FLOAT", microBatchShape};

  std::vector<int64_t> batchShape{batchSize, sampleDim};
  TensorInfo batchInfo{"FLOAT", microBatchShape};

  std::vector<int64_t> sampleShape{sampleDim};
  TensorInfo sampleInfo{"FLOAT", sampleShape};

  std::vector<int64_t> stepDataShape{stepSize, batchSize, sampleDim};
  TensorInfo stepDataInfo{"FLOAT", stepDataShape};

  auto input0 = builder->addInputTensor(batchInfo, "0tupni");

  WeightsIO weightsRead;

  std::vector<float> weight0(sampleDim, 100.0f);
  std::vector<float> rb0(sampleDim, -777.0f);
  ConstVoidData cvd0({weight0.data(), sampleInfo});
  auto w0Id = builder->addInitializedInputTensor(cvd0, w0name);
  weightsRead.insert(w0Id, {rb0.data(), sampleInfo});

  std::vector<float> weight1(sampleDim, 200.0f);
  std::vector<float> rb1(sampleDim, -777.0f);
  ConstVoidData cvd1({weight1.data(), sampleInfo});
  auto w1Id = builder->addInitializedInputTensor(cvd1, w1name);
  weightsRead.insert(w1Id, {rb1.data(), sampleInfo});

  auto add0 = aiOnnx.add({w0Id, input0});
  auto add1 = aiOnnx.add({w1Id, add0});
  builder->addOutputTensor(add1);

  auto proto    = builder->getModelProto();
  auto dataFlow = DataFlow(stepSize, {});

  float lambda = 1.0;
  auto loss    = std::unique_ptr<Loss>(
      new L1Loss(add1, "l1LossVal", lambda, ReductionType::SUM));

  SessionOptions userOptions;
  std::map<std::string, std::string> deviceOpts{{"numIPUs", "1"}};

  if (gradAccl) {
    userOptions.enableGradientAccumulation = true;
    userOptions.accumulationFactor         = 2;
  }

  auto device =
      DeviceManager::createDeviceManager().createIpuModelDevice(deviceOpts);

  auto session = popart::TrainingSession::createFromOnnxModel(
      proto,
      dataFlow,
      {loss.get()},
      opt0, // construct with opt0, will switch to opt1, opt2 later
      device,
      InputShapeInfo(),
      userOptions,
      popart::Patterns(PatternsLevel::DEFAULT));

  session->prepareDevice();
  std::vector<float> v_input_x(stepDataInfo.nelms(), 3.1415);

  popart::NDArrayWrapper<float> input_x_wrapper(v_input_x.data(), stepDataInfo);

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

  // run 3 with opt2
  session->updateOptimizer(&opt2);
  session->optimizerFromHost();
  session->run(stepio);

  // read final weights back
  session->weightsToHost();
  session->readWeights(weightsRead);

  return std::array<float, 2>{rb0[0], rb1[0]};
}

// -----------
BOOST_AUTO_TEST_CASE(SgdMixedModeTestCpp1_0) {

  float lrTest0 = 1.0f / 4.0f;
  popart::SGD opt0({
      {"defaultLearningRate", {lrTest0, true}},
      {"defaultMomentum", {1.0f, true}},
  });
  auto opt1 = opt0;
  auto opt2 = opt0;

  // using the pytorch equations for a weight W0:
  //
  // (1) g = g + wd * w
  // (2) v = v * mm + (1 - dp) * g
  // (3) w = w - lr * v

  // (t) g,v,w =  -1,0,W0
  // (1) g,v,w =  -1,0,W0
  // (2) g,v,w =  -1,-1, W0
  // (3) g,v,w =  -1,-1,W0-lr
  //
  // (t) g,v,w =  -1,-1,W0-lr
  // (1) g,v,w =  -1,-1,W0-lr
  // (2) g,v,w =  -1,-2,W0-lr
  // (3) g,v,w =  -1,-2,W0-3*lr
  //
  // (t) g,v,w =  -1,-2,W0-3*lr
  // (1) g,v,w =  -1,-2,W0-3*lr
  // (2) g,v,w =  -1,-3,W0-3*lr
  // (3) g,v,w =  -1,-3,W0-6*lr

  auto results = getResults(opt0, opt1, opt2);
  // check that expected values are the same as read-back values
  auto absdiff0 = getAbsDiff(100 - 6 * lrTest0, std::get<0>(results));
  BOOST_CHECK(absdiff0 < 1e-5f);
  auto absdiff1 = getAbsDiff(200 - 6 * lrTest0, std::get<1>(results));
  BOOST_CHECK(absdiff0 < 1e-5f);
}

BOOST_AUTO_TEST_CASE(SgdMixedModeTestCpp1_1) {
  // as test case 0, but with loss scaling and velocity scaling

  float lrTest0 = 1.0f / 4.0f;
  popart::SGD opt0({
      {"defaultLearningRate", {lrTest0, true}},
      {"defaultMomentum", {1.0f, true}},
      {"defaultVelocityScaling", {14.15f, true}},
      {"lossScaling", {0.15f, true}},
  });
  auto opt1     = opt0;
  auto opt2     = opt0;
  auto results  = getResults(opt0, opt1, opt2);
  auto absdiff0 = getAbsDiff(100 - 6 * lrTest0, std::get<0>(results));
  BOOST_CHECK(absdiff0 < 1e-5f);
  auto absdiff1 = getAbsDiff(200 - 6 * lrTest0, std::get<1>(results));
  BOOST_CHECK(absdiff0 < 1e-5f);
}

BOOST_AUTO_TEST_CASE(SgdMixedModeTestCpp1_2) {
  // as test case 1, but non-const for all

  float lrTest0 = 1.0f / 4.0f;
  popart::SGD opt0({
      {"defaultLearningRate", {lrTest0, false}},
      {"defaultMomentum", {1.0f, false}},
      {"defaultVelocityScaling", {14.15f, false}},
      {"lossScaling", {0.15f, false}},
  });
  auto opt1     = opt0;
  auto opt2     = opt0;
  auto results  = getResults(opt0, opt1, opt2);
  auto absdiff0 = getAbsDiff(100 - 6 * lrTest0, std::get<0>(results));
  BOOST_CHECK(absdiff0 < 1e-5f);
  auto absdiff1 = getAbsDiff(200 - 6 * lrTest0, std::get<1>(results));
  BOOST_CHECK(absdiff0 < 1e-5f);
}

BOOST_AUTO_TEST_CASE(SgdMixedModeTestCpp1_3) {

  // Caveat about updating SGD and comparing to pytorch:
  // ---------------------------------
  // Pytorch effectively does:
  //
  //    v <- v * mm  + (1 - dp) * wd * w
  //    v <- v + (1 - dp) * g
  //    w <- w - lr * v
  // x
  //    v <- v * mm  + (1 - dp) * wd * w
  //    v <- v + (1 - dp) * g
  //    w <- w - lr * v
  // x
  //    v <- v * mm  + (1 - dp) * wd * w
  //    v <- v + (1 - dp) * g
  //    w <- w - lr * v
  // x
  // ...
  //
  // Our implementation is the same, but the optimizer value updates do not
  // come at the x's. We do:
  //
  // Host side initialization with :
  //    v <- (1 - dp) * wd * w,
  //
  //  and then we repeat:
  //    v <- v + (1 - dp) * g
  //    w <- w - lr * v
  //    v <- v * mm  + (1 - dp) * wd * w
  // y
  //    v <- v + (1 - dp) * g
  //    w <- w - lr * v
  //    v <- v * mm  + (1 - dp) * wd * w
  // y
  //    v <- v + (1 - dp) * g
  //    w <- w - lr * v
  //    v <- v * mm  + (1 - dp) * wd * w
  // y
  //
  // The update of optimizer parameters takes place at the 'y's.
  //
  // So there is a lag, which will be noticed if mm, dp, and wd are updated;
  // but not if lr is.

  float wd0 = 0.1;
  float mm0 = 0.3;
  float dp0 = 0.4;
  float lr0 = 0.2;

  float wd1 = 0.1;
  float mm1 = 0.3;
  float dp1 = 0.4;
  float lr1 = 0.2; // learning rate can change and still be identical to pytorch

  float wd2 = 0.1;
  float mm2 = 0.3;
  float dp2 = 0.4;
  float lr2 = 0.3;

  popart::SGD opt0({{"defaultDampening", {dp0, true}},
                    {"defaultLearningRate", {lr0, false}},
                    {"defaultWeightDecay", {wd0, true}},
                    {"defaultMomentum", {mm0, false}}});

  popart::SGD opt1({{"defaultDampening", {dp1, true}},
                    {"defaultLearningRate", {lr1, false}},
                    {"defaultWeightDecay", {wd1, true}},
                    {"defaultMomentum", {mm1, false}}});

  popart::SGD opt2({{"defaultDampening", {dp2, true}},
                    {"defaultLearningRate", {lr2, false}},
                    {"defaultWeightDecay", {wd2, true}},
                    {"defaultMomentum", {mm2, false}}});

  float w0star{100};
  float g0star{0};
  float v0star{0};
  pytorchUpdate(w0star, g0star, v0star, wd0, mm0, dp0, lr0);
  pytorchUpdate(w0star, g0star, v0star, wd1, mm1, dp1, lr1);
  pytorchUpdate(w0star, g0star, v0star, wd2, mm2, dp2, lr2);

  float w1star{200};
  float g1star{0};
  float v1star{0};
  pytorchUpdate(w1star, g1star, v1star, wd0, mm0, dp0, lr0);
  pytorchUpdate(w1star, g1star, v1star, wd1, mm1, dp1, lr1);
  pytorchUpdate(w1star, g1star, v1star, wd2, mm2, dp2, lr2);

  auto results  = getResults(opt0, opt1, opt2);
  auto absdiff0 = getAbsDiff(w0star, std::get<0>(results));
  BOOST_CHECK(absdiff0 < 1e-5f);
  auto absdiff1 = getAbsDiff(w1star, std::get<1>(results));
  BOOST_CHECK(absdiff0 < 1e-5f);
}

BOOST_AUTO_TEST_CASE(SgdMixedModeTestCpp1_4) {

  // Testing that results match pytorch with the caveat discussed in test 3.

  float wd0 = 0.4f;
  float mm0 = 0.6f;
  float dp0 = 0.1f;
  float lr0 = 0.2f;
  float ls0 = 10.0f;

  float wd1 = 0.15f;
  float mm1 = 0.35f;
  float dp1 = 0.45f;
  float lr1 = 0.55f;
  float ls1 = 8.0f;

  float wd2 = 0.03f;
  float mm2 = 0.83f;
  float dp2 = 0.13f;
  float lr2 = 0.25f;
  float ls2 = 6.0f;

  // Note: see test 5 for velocity scaling changing during training
  float vs = 3.0f;

  popart::SGD opt0({{"defaultDampening", {dp0, false}},
                    {"defaultLearningRate", {lr0, false}},
                    {"defaultWeightDecay", {wd0, false}},
                    {"defaultVelocityScaling", {vs, true}},
                    {"lossScaling", {ls0, false}},
                    {"defaultMomentum", {mm0, false}}});

  popart::SGD opt1({{"defaultDampening", {dp1, false}},
                    {"defaultLearningRate", {lr1, false}},
                    {"defaultWeightDecay", {wd1, false}},
                    {"defaultVelocityScaling", {vs, true}},
                    {"lossScaling", {ls1, false}},
                    {"defaultMomentum", {mm1, false}}});

  popart::SGD opt2({{"defaultDampening", {dp2, false}},
                    {"defaultLearningRate", {lr2, false}},
                    {"defaultWeightDecay", {wd2, false}},
                    {"defaultVelocityScaling", {vs, true}},
                    {"lossScaling", {ls2, false}},
                    {"defaultMomentum", {mm2, false}}});

  auto getInitialV = [](float dp, float wd, float W) {
    // note that we do not include the velocity scaling term here, pytorch
    // updates are independent of velocity scaling.
    return (1.0f - dp) * wd * W;
  };

  float w0star = 100;
  float g0star = 0;
  float v0star = getInitialV(dp0, wd0, w0star);
  laggedPytorchUpdate(w0star, g0star, v0star, wd0, mm0, dp0, lr0);
  laggedPytorchUpdate(w0star, g0star, v0star, wd1, mm1, dp1, lr1);
  laggedPytorchUpdate(w0star, g0star, v0star, wd2, mm2, dp2, lr2);

  float w1star = 200;
  float g1star = 0;
  float v1star = getInitialV(dp0, wd0, w1star);
  laggedPytorchUpdate(w1star, g1star, v1star, wd0, mm0, dp0, lr0);
  laggedPytorchUpdate(w1star, g1star, v1star, wd1, mm1, dp1, lr1);
  laggedPytorchUpdate(w1star, g1star, v1star, wd2, mm2, dp2, lr2);

  auto results  = getResults(opt0, opt1, opt2);
  auto absdiff0 = getAbsDiff(w0star, std::get<0>(results));
  BOOST_CHECK(absdiff0 < 1e-5f);
  auto absdiff1 = getAbsDiff(w1star, std::get<1>(results));
  BOOST_CHECK(absdiff0 < 1e-5f);
}

BOOST_AUTO_TEST_CASE(SgdMixedModeTestCpp1_5) {

  // As in test 4, but now with velocity scaling term changing.

  float wd0 = 0.4f;
  float mm0 = 0.6f;
  float dp0 = 0.1f;
  float lr0 = 0.2f;
  float ls0 = 10.0f;
  float vs0 = 3.0f;

  float wd1 = 0.15f;
  float mm1 = 0.35f;
  float dp1 = 0.45f;
  float lr1 = 0.55f;
  float ls1 = 8.0f;
  float vs1 = 6.0f;

  float wd2 = 0.03f;
  float mm2 = 0.83f;
  float dp2 = 0.13f;
  float lr2 = 0.25f;
  float ls2 = 6.0f;
  float vs2 = 9.0f;

  popart::SGD opt0({{"defaultDampening", {dp0, false}},
                    {"defaultLearningRate", {lr0, false}},
                    {"defaultWeightDecay", {wd0, false}},
                    {"defaultVelocityScaling", {vs0, false}},
                    {"lossScaling", {ls0, false}},
                    {"defaultMomentum", {mm0, false}}});

  popart::SGD opt1({{"defaultDampening", {dp1, false}},
                    {"defaultLearningRate", {lr1, false}},
                    {"defaultWeightDecay", {wd1, false}},
                    {"defaultVelocityScaling", {vs1, false}},
                    {"lossScaling", {ls1, false}},
                    {"defaultMomentum", {mm1, false}}});

  popart::SGD opt2({{"defaultDampening", {dp2, false}},
                    {"defaultLearningRate", {lr2, false}},
                    {"defaultWeightDecay", {wd2, false}},
                    {"defaultVelocityScaling", {vs2, false}},
                    {"lossScaling", {ls2, false}},
                    {"defaultMomentum", {mm2, false}}});

  auto getInitialV = [](float dp, float wd, float vs, float W) {
    return (1.0f - dp) * wd * vs * W;
  };

  float w0star = 100;
  float g0star = 0;
  float v0star = getInitialV(dp0, wd0, vs0, w0star);
  laggedPytorchUpdateWithScaling(
      w0star, g0star, v0star, wd0, mm0, dp0, lr0, vs0, ls0);
  laggedPytorchUpdateWithScaling(
      w0star, g0star, v0star, wd1, mm1, dp1, lr1, vs1, ls1);
  laggedPytorchUpdateWithScaling(
      w0star, g0star, v0star, wd2, mm2, dp2, lr2, vs2, ls2);

  float w1star = 200;
  float g1star = 0;
  float v1star = getInitialV(dp0, wd0, vs0, w1star);
  laggedPytorchUpdateWithScaling(
      w1star, g1star, v1star, wd0, mm0, dp0, lr0, vs0, ls0);
  laggedPytorchUpdateWithScaling(
      w1star, g1star, v1star, wd1, mm1, dp1, lr1, vs1, ls1);
  laggedPytorchUpdateWithScaling(
      w1star, g1star, v1star, wd2, mm2, dp2, lr2, vs2, ls2);

  auto results  = getResults(opt0, opt1, opt2);
  auto absdiff0 = getAbsDiff(w0star, std::get<0>(results));
  BOOST_CHECK(absdiff0 < 1e-5f);
  auto absdiff1 = getAbsDiff(w1star, std::get<1>(results));
  BOOST_CHECK(absdiff0 < 1e-5f);
}

BOOST_AUTO_TEST_CASE(SgdMixedModeTestCpp1_6) {

  // As in test 4, but decouple the 2 weights by using insertSpecific

  // const weight decay, different on the 2 weights
  float defaultWd = 0.2f;
  float weight1Wd = 0.3f;
  // variable default momentum, different on the 2 weights
  float defaultMm0 = 0.9f;
  float defaultMm1 = 0.6f;
  float defaultMm2 = 0.5f;
  float weight0Mm  = 0.85f; // constant momentum for weight 0
  // constant dampening, the same on the 2 weights
  float dp = 0.2f;
  // variable learning rate, different on the 2 weights
  float defaultLr0 = 1.0;
  float defaultLr1 = 0.9;
  float defaultLr2 = 0.8;
  float weight0Lr0 = 0.7;
  float weight0Lr1 = 0.6;
  float weight0Lr2 = 0.4;
  // constant loss scaling
  float ls = 600.0f;
  // constant velocity scaling, different on the 2 weights
  float defaultVs = 0.1;
  float weight0Vs = 0.01;

  popart::SGD opt0({{"defaultDampening", {dp, true}},
                    {"defaultLearningRate", {defaultLr0, false}},
                    {"defaultWeightDecay", {defaultWd, true}},
                    {"defaultVelocityScaling", {defaultVs, true}},
                    {"lossScaling", {ls, true}},
                    {"defaultMomentum", {defaultMm0, false}}});

  // all values without a key in insertSpecific will take default values above
  opt0.insertSpecific(w0name,
                      {{"momentum", {weight0Mm, true}},
                       {"learningRate", {weight0Lr0, false}},
                       {"velocityScaling", {weight0Vs, true}}});
  opt0.insertSpecific(w1name, {{"weightDecay", {weight1Wd, true}}});

  popart::SGD opt1({{"defaultDampening", {dp, true}},
                    {"defaultLearningRate", {defaultLr1, false}},
                    {"defaultWeightDecay", {defaultWd, true}},
                    {"defaultVelocityScaling", {defaultVs, true}},
                    {"lossScaling", {ls, true}},
                    {"defaultMomentum", {defaultMm1, false}}});
  opt1.insertSpecific(w0name,
                      {{"momentum", {weight0Mm, true}},
                       {"learningRate", {weight0Lr1, false}},
                       {"velocityScaling", {weight0Vs, true}}});
  opt1.insertSpecific(w1name, {{"weightDecay", {weight1Wd, true}}});

  popart::SGD opt2({{"defaultDampening", {dp, true}},
                    {"defaultLearningRate", {defaultLr2, false}},
                    {"defaultWeightDecay", {defaultWd, true}},
                    {"defaultVelocityScaling", {defaultVs, true}},
                    {"lossScaling", {ls, true}},
                    {"defaultMomentum", {defaultMm2, false}}});
  opt2.insertSpecific(w0name,
                      {{"momentum", {weight0Mm, true}},
                       {"learningRate", {weight0Lr2, false}},
                       {"velocityScaling", {weight0Vs, true}}});
  opt2.insertSpecific(w1name, {{"weightDecay", {weight1Wd, true}}});

  auto getInitialV = [](float dp, float wd, float W) {
    // no need to include vs, it is constant so we can use vanilla pytorch
    // update equations
    return (1.0f - dp) * wd * W;
  };

  float w0star = 100;
  float g0star = 0;
  float v0star = getInitialV(dp, defaultWd, w0star);
  laggedPytorchUpdate(
      w0star, g0star, v0star, defaultWd, weight0Mm, dp, weight0Lr0);
  laggedPytorchUpdate(
      w0star, g0star, v0star, defaultWd, weight0Mm, dp, weight0Lr1);
  laggedPytorchUpdate(
      w0star, g0star, v0star, defaultWd, weight0Mm, dp, weight0Lr2);

  float w1star = 200;
  float g1star = 0;
  float v1star = getInitialV(dp, weight1Wd, w1star);
  laggedPytorchUpdate(
      w1star, g1star, v1star, weight1Wd, defaultMm0, dp, defaultLr0);
  laggedPytorchUpdate(
      w1star, g1star, v1star, weight1Wd, defaultMm1, dp, defaultLr1);
  laggedPytorchUpdate(
      w1star, g1star, v1star, weight1Wd, defaultMm2, dp, defaultLr2);

  auto results  = getResults(opt0, opt1, opt2);
  auto absdiff0 = getAbsDiff(w0star, std::get<0>(results));
  BOOST_CHECK(absdiff0 < 1e-5f);
  auto absdiff1 = getAbsDiff(w1star, std::get<1>(results));
  BOOST_CHECK(absdiff0 < 1e-5f);
}

BOOST_AUTO_TEST_CASE(SgdMixedModeTestCpp1_7) {
  // as test case 2, but with gradient accumulation

  float lrTest0 = 1.0f / 4.0f;
  popart::SGD opt0({
      {"defaultLearningRate", {lrTest0, false}},
      {"defaultMomentum", {1.0f, false}},
      {"defaultVelocityScaling", {14.15f, false}},
      {"lossScaling", {0.15f, false}},
  });
  auto opt1    = opt0;
  auto opt2    = opt0;
  auto results = getResults(opt0, opt1, opt2, true);
  // including factor 2 for accumulation factor
  auto absdiff0 = getAbsDiff(100 - 2 * 6 * lrTest0, std::get<0>(results));
  BOOST_CHECK(absdiff0 < 1e-5f);
  auto absdiff1 = getAbsDiff(200 - 2 * 6 * lrTest0, std::get<1>(results));
  BOOST_CHECK(absdiff0 < 1e-5f);
}
