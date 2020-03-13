// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE NumericsInplaceVsNot1Test

#include <climits>
#include <cmath>
#include <random>
#include <utility>

#include <boost/filesystem.hpp>
#include <boost/test/unit_test.hpp>

#include <onnx/checker.h>

#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/devicemanager.hpp>
#include <popart/filereader.hpp>
#include <popart/iarray.hpp>
#include <popart/inputshapeinfo.hpp>
#include <popart/names.hpp>
#include <popart/ndarraywrapper.hpp>
#include <popart/ndindices.hpp>
#include <popart/patterns/patterns.hpp>
#include <popart/session.hpp>
#include <popart/sessionoptions.hpp>

using namespace popart;

// Implements the following pseudocode:

// activeTensorSet = { Tensor(16,16) }
// targetSetSize = 10.
// nOpsToCreate = 100.

// for i = 0 : nOpsToCreate
//   if active.size() < targetSetSize:
//     tensor = choose random Tensor from activeTensorSet
//     scaleFactor = choose randomly from {0.5, 1.0, 2.0}
//     scaled = tensor*scaleFactor
//     slice0, slice1 = slice scaled on random axis (0 or 1) at random index
//     (0->15) concated = concatenate(slice1, slice0) active.insert(concated)
//   else:
//    choose 2 from active, remove them from active, and insert their sum into
//    active.

enum class RunMode { SCALE = 0, SCALE_IN_PLACE, MUL };

BOOST_AUTO_TEST_CASE(Inplace_numericsIpNip1) {

  float perturbSize = 1e-3F;

  auto getValue = [perturbSize](const RunMode runMode,
                                int seed,
                                bool outlining,
                                bool perturbInput) {
    using namespace std::chrono;
    auto t0 = steady_clock::now();

    constexpr unsigned int H = 16;
    constexpr unsigned int W = 16;

    constexpr unsigned int targetSetSize  = 8;
    constexpr unsigned int numOpsToCreate = 30;

    Shape inShape{H, W};

    // see T15121: exponential slow down in a popx algorithm.
    constexpr bool useInitialReductionToAvoidLayoutSearch{true};
    if (useInitialReductionToAvoidLayoutSearch) {
      int64_t redFactor = 3;
      inShape.push_back(redFactor);
    }

    // Build an onnx model
    auto builder     = Builder::create();
    auto aiOnnx      = builder->aiOnnxOpset9();
    auto aiGraphcore = builder->aiGraphcoreOpset1();

    TensorInfo inInfo("FLOAT", inShape);
    auto inTensor    = builder->addInputTensor(inInfo);
    auto firstTensor = inTensor;
    if (useInitialReductionToAvoidLayoutSearch) {
      firstTensor = aiOnnx.reducesum({firstTensor}, {2}, false);
    }

    std::vector<TensorId> activeTensors;
    activeTensors.push_back(firstTensor);

    // Create scale factors as constant tensors
    const std::array<float, 3> scaleFactors{0.5, 1.0, 2.0};
    std::vector<TensorId> scaleFactorTensors;

    for (int i = 0; i < scaleFactors.size(); i++) {
      ConstVoidData scaleData = {&(scaleFactors[i]), {"FLOAT", Shape({})}};
      scaleFactorTensors.push_back(aiOnnx.constant(
          scaleData, std::string("scaleFactor") + std::to_string(i)));
    }

    // Use seeded random number generators
    std::default_random_engine eng(seed);
    std::uniform_int_distribution<unsigned int> dis;
    std::uniform_real_distribution<float> fdisPref(0, +10.0);

    // Function for adding a new op
    auto addNewOp = [H,
                     W,
                     &scaleFactors,
                     &scaleFactorTensors,
                     &builder,
                     &runMode,
                     &eng,
                     &dis,
                     &fdisPref,
                     &aiOnnx,
                     &aiGraphcore,
                     &activeTensors]() {
      if (activeTensors.size() < targetSetSize) {
        const auto &randomTensor =
            activeTensors[dis(eng) % activeTensors.size()];

        TensorId scaledTensor;

        if (runMode == RunMode::MUL) {
          auto randScl =
              scaleFactorTensors[dis(eng) % scaleFactorTensors.size()];
          scaledTensor = aiOnnx.mul({randomTensor, randScl});
        } else {
          assert(runMode == RunMode::SCALE ||
                 runMode == RunMode::SCALE_IN_PLACE);
          auto randScl = scaleFactors[dis(eng) % scaleFactorTensors.size()];
          scaledTensor = aiGraphcore.scale({randomTensor}, randScl);
          builder->setInplacePreferences(
              scaledTensor, {{"ScaleInplace", 100.0f + fdisPref(eng)}});
        }

        unsigned int axis      = dis(eng) % 2;
        unsigned int axis_max  = (axis == 0) ? H : W;
        unsigned int slice_idx = (dis(eng) % (axis_max - 2)) + 1;

        // Allow some overlap
        unsigned int overlap = dis(eng) % slice_idx;

        auto slicedTensor_0 = aiOnnx.slice({scaledTensor},
                                           {slice_idx}, // end
                                           {0},         // start
                                           {axis});

        auto slicedTensor_1 = aiOnnx.slice({scaledTensor},
                                           {axis_max - overlap},  // end
                                           {slice_idx - overlap}, // start
                                           {axis});

        auto concat = aiOnnx.concat({slicedTensor_1, slicedTensor_0}, axis);
        activeTensors.push_back(concat);
      } else {
        auto first_idx  = dis(eng) % activeTensors.size();
        auto second_idx = dis(eng) % activeTensors.size();

        if (second_idx < first_idx) {
          std::swap(second_idx, first_idx);
        }

        auto sum =
            aiOnnx.add({activeTensors[first_idx], activeTensors[second_idx]});

        activeTensors[first_idx] = activeTensors.back();

        if (second_idx != first_idx) {
          activeTensors.pop_back();
          activeTensors[second_idx] = activeTensors.back();
        }
        activeTensors.back() = sum;
      }
    };

    for (unsigned int i = 0; i < numOpsToCreate; i++) {
      addNewOp();
    }

    // Finally sum all tensors
    while (activeTensors.size() > 1) {
      activeTensors[0] = aiOnnx.add({activeTensors[0], activeTensors.back()});
      activeTensors.pop_back();
    }
    auto finalSum = activeTensors[0];

    auto out = aiOnnx.reducesum({finalSum}, {0, 1}, false);
    builder->addOutputTensor(out);

    auto proto      = builder->getModelProto();
    auto modelProto = io::getModelFromString(proto);

    // onnx::checker::check_model(modelProto);

    // Create the IR, adding outId as an anchor
    auto art      = AnchorReturnType("ALL");
    auto dataFlow = DataFlow(1, {{finalSum, art}});

    auto opts = SessionOptions();
    // opts.dotChecks.insert(DotCheck::BWD0);
    // opts.dotChecks.insert(DotCheck::FWD0);
    // opts.dotChecks.insert(DotCheck::FINAL);
    // opts.dotOpNames      = false;
    // opts.logDir          = "./dotfiles";
    // if (!boost::filesystem::exists(opts.logDir)) {
    //  boost::filesystem::create_directory(opts.logDir);
    // }

    opts.enableOutlining = outlining;
    if (outlining) {
      opts.outlineThreshold = 1e-7f;
    }

    auto cpuDevice =
        popart::DeviceManager::createDeviceManager().createCpuDevice();

    auto session = popart::InferenceSession::createFromOnnxModel(
        proto,
        dataFlow,
        cpuDevice,
        {},
        popart::InputShapeInfo(),
        opts,
        popart::Patterns(PatternsLevel::NONE)
            .enableInPlace(runMode == RunMode::SCALE_IN_PLACE));

    // prepare the anchors
    std::vector<float> rawOutputData(H * W, 0);
    Shape outShape{H, W};
    popart::NDArrayWrapper<float> outData(rawOutputData.data(), outShape);

    std::map<popart::TensorId, popart::IArray &> anchors = {
        {finalSum, outData},
    };

    session->prepareDevice();

    // generate random input data in the range [0 )
    std::uniform_int_distribution<int> fdisInit(-5, +5);
    float perturbFactor = perturbInput ? perturbSize : 0.0;
    std::vector<float> vInData(inInfo.nelms(), 0);

    for (uint64_t i = 0; i < inInfo.nelms(); ++i) {
      vInData[i] = static_cast<float>(std::pow(2.0, fdisInit(eng)));

      // Ensure it actually is a power of 2
      int exp;
      assert(frexp(vInData[i], &exp) == 0.5F);

      vInData[i] += perturbFactor * fdisInit(eng);
    }

    popart::NDArrayWrapper<float> inData(vInData.data(), inShape);
    std::map<popart::TensorId, popart::IArray &> inputs = {{inTensor, inData}};

    popart::StepIO stepio(inputs, anchors);
    session->run(stepio);

    auto t1 = steady_clock::now();
    auto dX = duration<double, std::milli>(t1 - t0).count();
    std::cout << "seed=" << seed << "  outline=" << outlining << "  time=" << dX
              << " [ms]" << std::endl;

    return rawOutputData[0];
  };

  auto runTest = [&getValue, &perturbSize](int seed) {
    // scale
    auto v000 = getValue(RunMode::SCALE, seed, false, false);
    // scale+outline
    auto v010 = getValue(RunMode::SCALE, seed, true, false);
    // scale+inplace
    auto v100 = getValue(RunMode::SCALE_IN_PLACE, seed, false, false);
    // scale+inplace+outline
    auto v110 = getValue(RunMode::SCALE_IN_PLACE, seed, true, false);
    // mul
    auto v200 = getValue(RunMode::SCALE_IN_PLACE, seed, false, false);
    // mul+outline
    auto v210 = getValue(RunMode::SCALE_IN_PLACE, seed, true, false);

    // scale+perturb
    auto v001 = getValue(RunMode::SCALE, seed, false, true);

    std::cout << std::scientific << "Final value with Vanilla is " << v000
              << ". Some discrepencies : "
              << "\nScale Inplace to Scale : " << std::fabs(v100 - v000)
              << "\nOutline to Scale : " << std::fabs(v010 - v000)
              << "\nScale Inplace to Outline : " << std::fabs(v110 - v010)
              << "\nScale Inplace and outline to Scale : "
              << std::fabs(v110 - v000)
              << "\nScale Inplace to Outline : " << std::fabs(v110 - v010)
              << "\nScale Inplace and outline to Scale : "
              << std::fabs(v110 - v000)
              << "\nScale to Scale perturbed: " << std::fabs(v000 - v001)
              << std::endl;

    // inplacing, no outlining:
    BOOST_CHECK(std::fabs(v100 - v000) == 0.0F);
    // inplacing and outlining:
    BOOST_CHECK(std::fabs(v110 - v000) == 0.0F);
    // mul
    BOOST_CHECK(std::fabs(v200 - v000) == 0.0F);
    // mul and outlining
    BOOST_CHECK(std::fabs(v210 - v000) == 0.0F);

    // this is not a useless test, permute input has an effect.
    BOOST_CHECK(std::fabs(v001 - v000) > perturbSize);
  };

  const int seed = 18;
  runTest(seed);
}
