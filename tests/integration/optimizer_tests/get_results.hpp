// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <array>
#include <boost/test/unit_test.hpp>
#include <iostream>
#include <map>
#include <random>
#include <tuple>
#include <vector>

#define protected public
#include <filereader.hpp>
#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/devicemanager.hpp>
#include <popart/error.hpp>
#include <popart/half.hpp>
#include <popart/inputshapeinfo.hpp>
#include <popart/ndarraywrapper.hpp>
#include <popart/op/identity.hpp>
#include <popart/op/ipucopy.hpp>
#include <popart/op/l1.hpp>
#include <popart/op/restore.hpp>
#include <popart/op/sgd0varupdate.hpp>
#include <popart/op/stash.hpp>
#include <popart/session.hpp>
#include <popart/sgd.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/tensornames.hpp>
#undef protected

// TODO T12595 : put the implemenations below into a .cpp file

// names of 2 weights used in model
constexpr const char *w0name = "__w0__";
constexpr const char *w1name = "__w1__";

// Used in model. Exposed because some tests directly compute the updates
// instead of emulating them with the helpers.
int64_t replicationFactor  = 2;
int64_t accumulationFactor = 5;

namespace _detail {

// TODO T10881 : cleaner solution to acquisition of IPU failure
std::array<float, 2> acquisitionFailure{-99.0f, -99.0f};

} // namespace _detail

// Pytorch update equations with loss and velocity scaling added. Required when
// the velocity-scaled v tensor must be modelled.
void pytorchUpdateWithScaling(float &w,
                              float &g,
                              float &v,
                              float wd,
                              float mm,
                              float dp,
                              float lr,
                              bool repl  = false,
                              bool accum = false,
                              float vs   = 1.0f,
                              float ls   = 1.0f) {
  // from the model below
  g = +1.0f * ls;
  if (repl) {
    g *= static_cast<float>(replicationFactor);
  }
  if (accum) {
    g *= static_cast<float>(accumulationFactor);
  }

  v = v * mm + (1.0f - dp) * vs / ls * g + (1.0f - dp) * vs * wd * w;

  w -= lr / vs * v;
}

// The pytorch update equations:
//
// (1) g = g + wd * w
// (2) v = v * mm + (1 - dp) * g
// (3) w = w - lr * v
//
// Note there is no velocity or loss scaling in the Pytorch update, but the
// final w should not affected by this anyway. v is affected however, so
// pytorchUpdateWithScaling should be used to model v under velocity scaling.
void pytorchUpdate(float &w,
                   float &g,
                   float &v,
                   float wd,
                   float mm,
                   float dp,
                   float lr,
                   bool repl  = false,
                   bool accum = false) {
  return pytorchUpdateWithScaling(w, g, v, wd, mm, dp, lr, repl, accum, 1, 1);
}

namespace _detail {

template <typename Derived> struct SGDTestConfig {
  /*
    Must implement

    static constexpr popart::SGDAccumulatorAndMomentum sgdAccMm;
  */

  static float getInitialV(float dp, float wd, float vs, float W) {
    throw popart::internal_error(
        "Class {} that extends SGDTestConfig must implement the static "
        "getInitialV method.",
        typeid(Derived).name());
  }

  static float getInitialV(float dp, float wd, float W) {
    return Derived::getInitialV(dp, wd, 1.0f, W);
  }

  static void laggedPytorchUpdate(float &w,
                                  float &g,
                                  float &v,
                                  float wd,
                                  float mm,
                                  float dp,
                                  float lr,
                                  int64_t replFactor,
                                  int64_t acclFactor) {
    throw popart::internal_error(
        "Class {} that extends SGDTestConfig must implement the static "
        "laggedPytorchUpdate method.",
        typeid(Derived).name());
  }

  static void laggedPytorchUpdateWithScaling(float &w,
                                             float &g,
                                             float &v,
                                             float wd,
                                             float mm,
                                             float dp,
                                             float lr,
                                             float vs,
                                             float ls) {
    throw popart::internal_error(
        "Class {} that extends SGDTestConfig must implement the static"
        "laggedPytorchUpdateWithScaling method.",
        typeid(Derived).name());
  }
};

} // namespace _detail

struct SGD2TestConfig : public _detail::SGDTestConfig<SGD2TestConfig> {
  static constexpr popart::SGDAccumulatorAndMomentum sgdAccMm =
      popart::SGDAccumulatorAndMomentum::Separate;

  using _detail::SGDTestConfig<SGD2TestConfig>::getInitialV;

  static float getInitialV(float, float, float vs, float) { return vs * 0.0f; };

  static void laggedPytorchUpdate(float &w,
                                  float &g,
                                  float &v,
                                  float wd,
                                  float mm,
                                  float dp,
                                  float lr,
                                  bool repl  = false,
                                  bool accum = false) {
    pytorchUpdate(w, g, v, wd, mm, dp, lr, repl, accum);
  }

  static void laggedPytorchUpdateWithScaling(float &w,
                                             float &g,
                                             float &v,
                                             float wd,
                                             float mm,
                                             float dp,
                                             float lr,
                                             float vs,
                                             float ls) {
    pytorchUpdateWithScaling(w, g, v, wd, mm, dp, lr, false, false, vs, ls);
  }
};

struct SGD1TestConfig : public _detail::SGDTestConfig<SGD1TestConfig> {
  static constexpr popart::SGDAccumulatorAndMomentum sgdAccMm =
      popart::SGDAccumulatorAndMomentum::Combined;

  using _detail::SGDTestConfig<SGD1TestConfig>::getInitialV;

  static float getInitialV(float dp, float wd, float vs, float W) {
    return (1.0f - dp) * wd * vs * W;
  };

  static void laggedPytorchUpdate(float &w,
                                  float &g,
                                  float &v,
                                  float wd,
                                  float mm,
                                  float dp,
                                  float lr,
                                  bool repl  = false,
                                  bool accum = false) {

    // from the model below
    g = +1.0f;
    if (repl) {
      g *= static_cast<float>(replicationFactor);
    }
    if (accum) {
      g *= static_cast<float>(accumulationFactor);
    }

    v = v + (1.0f - dp) * g;
    w = w - lr * v;
    v = v * mm + (1.0f - dp) * wd * w;
  }

  static void laggedPytorchUpdateWithScaling(float &w,
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
};

using SGD1And2TestConfigs = std::tuple<SGD1TestConfig, SGD2TestConfig>;

bool acquisitionFailure(std::array<float, 2> res) {
  return res == _detail::acquisitionFailure;
}

float getAbsDiff(float expected, float observed) {
  float absv = std::abs(expected - observed);
  std::cout << "Expected=" << expected << ", observed=" << observed
            << " with absolute difference=" << absv << std::endl;
  return absv;
}

template <typename T> std::string getFloatString() {
  throw popart::error("unrecognised type");
}

template <> std::string getFloatString<float>() { return "FLOAT"; }
template <> std::string getFloatString<popart::float16_t>() {
  return "FLOAT16";
}

template <typename T>
std::array<float, 2>
getResults(const popart::SGD &opt0, // initial Optimizer
           const popart::SGD &opt1, // Optimizer to switch to
           const popart::SGD &opt2, // Last Optimizer
           bool gradAccl,
           bool graphRepl) {

  auto floatString = getFloatString<T>();

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

  int64_t sampleDim            = 1;
  int64_t samplesPerMicroBatch = 1;
  int64_t samplesPerBatch      = (graphRepl ? replicationFactor : 1) *
                            (gradAccl ? accumulationFactor : 1) *
                            samplesPerMicroBatch;
  int64_t batchesPerStep = 1;

  std::vector<int64_t> microBatchShape{samplesPerMicroBatch, sampleDim};
  TensorInfo microBatchInfo{floatString, microBatchShape};

  std::vector<int64_t> batchShape{samplesPerBatch, sampleDim};
  TensorInfo batchInfo{floatString, microBatchShape};

  std::vector<int64_t> sampleShape{sampleDim};
  TensorInfo sampleInfo{floatString, sampleShape};

  std::vector<int64_t> stepDataShape{
      batchesPerStep, samplesPerBatch, sampleDim};
  TensorInfo stepDataInfo{floatString, stepDataShape};

  auto input0 = builder->addInputTensor(microBatchInfo, "0tupni");

  WeightsIO weightsRead;

  std::vector<T> weight0(sampleDim, 100.0f);
  std::vector<T> rb0(sampleDim, -777.0f);
  ConstVoidData cvd0({weight0.data(), sampleInfo});
  auto w0Id = builder->addInitializedInputTensor(cvd0, w0name);
  assert(w0Id == w0name);
  weightsRead.insert(w0Id, {rb0.data(), sampleInfo});

  std::vector<T> weight1(sampleDim, 200.0f);
  std::vector<T> rb1(sampleDim, -777.0f);
  ConstVoidData cvd1({weight1.data(), sampleInfo});
  auto w1Id = builder->addInitializedInputTensor(cvd1, w1name);
  assert(w1Id == w1name);
  weightsRead.insert(w1Id, {rb1.data(), sampleInfo});

  auto add0 = aiOnnx.add({w0Id, input0});
  auto add1 = aiOnnx.add({w1Id, add0});
  auto l1 =
      builder->aiGraphcoreOpset1().l1loss({add1}, 1.0, ReductionType::Sum);

  auto proto    = builder->getModelProto();
  auto dataFlow = DataFlow(batchesPerStep);

  SessionOptions userOptions;
  std::map<std::string, std::string> deviceOpts{{"numIPUs", "1"}};

  userOptions.enableOutlining = false;

  if (gradAccl) {
    userOptions.enableGradientAccumulation = true;
    userOptions.accumulationFactor         = accumulationFactor;
  }

  if (graphRepl) {
    deviceOpts["numIPUs"]              = std::to_string(replicationFactor);
    userOptions.enableReplicatedGraphs = true;
    userOptions.replicatedGraphCount   = replicationFactor;
  }

  std::shared_ptr<DeviceInfo> device;
  if (!graphRepl) {
    device =
        DeviceManager::createDeviceManager().createIpuModelDevice(deviceOpts);
  } else {

    // TODO : cleaner acquisition failure solution T10881
    bool errorIfFailToAcquire = false;

    auto devices =
        popart::DeviceManager::createDeviceManager().enumerateDevices();
    if (devices.size() == 0) {
      if (errorIfFailToAcquire) {
        throw error("Failed to enumerate any devices in get_results.hpp");
      } else {
        return _detail::acquisitionFailure;
      }
    }

    device = DeviceManager::createDeviceManager().acquireAvailableDevice(
        replicationFactor);

    if (!device) {
      if (errorIfFailToAcquire) {
        throw error("Failed to acquire device in get_results.hpp");
      } else {
        return _detail::acquisitionFailure;
      }
    }
  }

  auto session = popart::TrainingSession::createFromOnnxModel(
      proto,
      dataFlow,
      l1,
      opt0, // construct with opt0, will switch to opt1, opt2 later
      device,
      InputShapeInfo(),
      userOptions,
      popart::Patterns(PatternsLevel::Default));

  session->prepareDevice();
  std::vector<T> v_input_x(stepDataInfo.nelms(), 3.1415);

  popart::NDArrayWrapper<T> input_x_wrapper(v_input_x.data(), stepDataInfo);

  std::map<popart::TensorId, popart::IArray &> inputs = {
      {input0, input_x_wrapper}};

  popart::StepIO stepio(inputs, {});

  session->weightsFromHost();

  // run 1 with opt0
  session->run(stepio);

  // run 2 with opt1
  session->updateOptimizerFromHost(&opt1);

  session->run(stepio);

  // run 3 with opt2
  session->updateOptimizerFromHost(&opt2);

  session->run(stepio);

  // read final weights back
  session->weightsToHost();
  session->readWeights(weightsRead);

  return std::array<float, 2>{static_cast<float>(rb0[0]),
                              static_cast<float>(rb1[0])};
}
