// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE NumericsInplaceVsNot0Test

#include <boost/filesystem.hpp>
#include <boost/test/unit_test.hpp>
#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/devicemanager.hpp>
#include <popart/filereader.hpp>
#include <popart/inputshapeinfo.hpp>
#include <popart/ir.hpp>
#include <popart/names.hpp>
#include <popart/ndarraywrapper.hpp>
#include <popart/op/l1.hpp>
#include <popart/optimizer.hpp>
#include <popart/session.hpp>
#include <popart/tensordata.hpp>
#include <popart/testdevice.hpp>

#include <chrono>
#include <complex>
#include <iostream>
#include <random>

using namespace popart;

// Module :
//
//
//   input :             [2^N, 2]
//                          |
//                      [Sigmoid] -------------------------------
//                          |                                    |
//           slice into 2^N slices of size [1,2]                 |
//                          |                                    |
//             [Random shuffle of indices]                       |   |
//                          |                                    |   |
//                          .                                    |   |
//            --------------------------------                   |   |
//            |  |  |                   |  |  |                  |   | Submod
//        for i = 0:(2^N)/2         for i = 0:(2^N)/2            |   | 3X
//          add slices 2*i           choose a slice              |   |
//           and 2*i + 1               at random                 |   |
//            |  |  |                   |  |  |                  |   |
//           [Sigmoid]               [Random Scale]              |   |
//           |   |  |                  |   |  |                  |   |
//           ----------------------------------                  |   |
//                           .                                   |
//                           |                                   |
//                       Tree Concat                             |
//                           |                                [Scale]
//                           |                                   |
//                           |                                   |
//                   ---------------------------------------------
//                                 |
//                               [Add]
//
//
//
// The Module is repeated J times, then there is a sum reduce:
//
//                    J repeats of the Module
//                             |
//                   ------------------
//                   |          |      |
//  in [2^N,2] -- Module -- Module -- Module -- ReduceSum

BOOST_AUTO_TEST_CASE(Inplace_numericsIpNip0) {

  float perturbSize = 1e-3;
  auto getValue     = [perturbSize](bool inp,
                                int seed,
                                int N,
                                int J,
                                bool outlining,
                                bool perturbInput) {
    using namespace std::chrono;
    auto t0 = steady_clock::now();

    Shape inShape{static_cast<int64_t>(std::pow(2, N)), 2};

    const bool useInitialReductionToAvoidLayoutSearch{false};
    if (useInitialReductionToAvoidLayoutSearch) {
      int64_t redFactor = 3;
      inShape.push_back(redFactor);
    }

    // The input is of shape (2^N, 2, redFactor)
    TensorInfo inInfo{"FLOAT", inShape};

    // generate random input data in the range [0, 1e-1)
    std::uniform_real_distribution<float> fdisInit(0, +1e-1);
    // possibly perturb each value by a value in [0, 1e-4)
    float perturbFactor = perturbInput ? perturbSize : 0.0;
    std::uniform_real_distribution<float> fdisScale(0.8, 1.3);
    std::uniform_real_distribution<float> fdisPref(0, +10.0);
    std::default_random_engine eng(seed);
    std::uniform_int_distribution<uint64_t> idis(
        0, std::numeric_limits<uint64_t>::max());

    std::vector<float> vInData(inInfo.nelms(), 0);
    for (uint64_t i = 0; i < inInfo.nelms(); ++i) {
      vInData[i] = fdisInit(eng);
      vInData[i] += perturbFactor * fdisInit(eng);
    }

    // Build an onnx model
    auto builder     = Builder::create();
    auto aiOnnx      = builder->aiOnnxOpset9();
    auto aiGraphcore = builder->aiGraphcoreOpset1();

    auto getSubmodule = [N,
                         &builder,
                         &eng,
                         &aiOnnx,
                         &aiGraphcore,
                         &fdisScale,
                         &fdisPref,
                         &idis](std::vector<TensorId> tensorsIn) {
      std::shuffle(tensorsIn.begin(), tensorsIn.end(), eng);
      std::vector<TensorId> tensorsOut;

      // the top half: add, sigmoid.
      for (int i = 0; i < std::pow(2, N - 1); ++i) {
        auto sum = aiOnnx.add({tensorsIn[2 * i], tensorsIn[2 * i + 1]});

        builder->setInplacePreferences(
            sum,
            {{"AddLhsInplace", 100.0f + fdisPref(eng)},
             {"AddRhsInplace", 100.0f + fdisPref(eng)}});

        auto sigmoidOut = aiOnnx.sigmoid({sum});
        builder->setInplacePreferences(
            sigmoidOut, {{"SigmoidInplace", 100.0f + fdisPref(eng)}});
        tensorsOut.push_back(sigmoidOut);
      }

      // the bottom half: scale.
      for (int i = 0; i < std::pow(2, N - 1); ++i) {
        TensorId id0  = tensorsIn[idis(eng) % tensorsIn.size()];
        auto scaleOut = aiGraphcore.scale({id0}, fdisScale(eng));
        builder->setInplacePreferences(
            scaleOut, {{"ScaleInplace", 100.0f + fdisPref(eng)}});
        tensorsOut.push_back(scaleOut);
      }
      return tensorsOut;
    };

    auto appendModule =
        [&builder, &eng, &fdisPref, &fdisScale, &aiOnnx, &getSubmodule](
            std::vector<TensorId> tensorsIn) {
          auto tensorsOut = tensorsIn;
          for (int i = 0; i < 3; ++i) {
            tensorsOut = getSubmodule(tensorsOut);
          }

          while (tensorsOut.size() != 1) {
            tensorsIn  = tensorsOut;
            tensorsOut = {};
            for (int i = 0; i < tensorsIn.size() / 2; ++i) {
              auto concat0 =
                  aiOnnx.concat({tensorsIn[2 * i], tensorsIn[2 * i + 1]}, 0);
              builder->setInplacePreferences(
                  concat0, {{"ConcatInplace", 100.0f + fdisPref(eng)}});
              tensorsOut.push_back(concat0);
            }
          }
          auto outCon = tensorsOut[0];

          return outCon;
        };

    auto getSliced = [N, &builder, &aiOnnx, &fdisPref, &eng](TensorId in) {
      std::vector<TensorId> slicedIds;
      for (int i = 0; i < std::pow(2, N); ++i) {

        auto sliceOut = aiOnnx.slice({in}, {i + 1, 2}, {i, 0}, {0, 1});
        builder->setInplacePreferences(
            sliceOut, {{"SliceInplace", 100.0f + fdisPref(eng)}});
        slicedIds.push_back(sliceOut);
      }
      return slicedIds;
    };

    auto inId = builder->addInputTensor(inInfo);

    TensorId singleTensor;
    if (useInitialReductionToAvoidLayoutSearch) {
      auto reducedOnFinal = aiOnnx.reducesum({inId}, {2}, false);
      singleTensor        = aiOnnx.sigmoid({reducedOnFinal});
    } else {
      singleTensor = aiOnnx.sigmoid({inId});
    }

    builder->setInplacePreferences(
        singleTensor, {{"SigmoidInplace", 100.0f + fdisPref(eng)}});

    for (int i = 0; i < J; ++i) {
      // break the input into many [1,2] tensors:
      auto slicedTensors  = getSliced(singleTensor);
      auto residualTensor = appendModule(slicedTensors);

      // the skip connect. We alternate between highest
      // priority and lowest priority
      auto skipTensor = aiGraphcore.scale({singleTensor}, fdisScale(eng));
      builder->setInplacePreferences(
          skipTensor,
          {{"ScaleInplace", 10.0f + 200.0f * (i % 2 == 1) + fdisPref(eng)}});

      singleTensor = aiOnnx.add({skipTensor, residualTensor});

      builder->setInplacePreferences(
          singleTensor,
          {{"AddLhsInplace", 100.0f + fdisPref(eng)},
           {"AddRhsInplace", 100.0f + fdisPref(eng)}});
    }

    auto out = aiOnnx.reducesum({singleTensor}, {0, 1}, false);
    builder->addOutputTensor(out);

    auto proto      = builder->getModelProto();
    auto modelProto = io::getModelFromString(proto);

    // Create the IR, adding outId as an anchor
    auto art      = AnchorReturnType("ALL");
    auto dataFlow = DataFlow(1, {{out, art}});

    auto opts = SessionOptions();
    opts.dotChecks.insert(DotCheck::BWD0);
    opts.dotChecks.insert(DotCheck::FWD0);
    opts.dotChecks.insert(DotCheck::FINAL);
    opts.dotOpNames      = false;
    opts.logDir          = "./dotfiles";
    opts.enableOutlining = outlining;
    if (outlining) {
      opts.outlineThreshold = 1e-7f;
    }
    if (!boost::filesystem::exists(opts.logDir)) {
      boost::filesystem::create_directory(opts.logDir);
    }

    auto device = popart::createTestDevice(TEST_TARGET);

    auto session = popart::InferenceSession::createFromOnnxModel(
        proto,
        dataFlow,
        device,
        {},
        popart::InputShapeInfo(),
        opts,
        popart::Patterns(PatternsLevel::NONE).enableInPlace(inp));

    // prepare the anchors
    float rawOutputData;
    Shape outShape{};
    popart::NDArrayWrapper<float> outData(&rawOutputData, outShape);

    std::map<popart::TensorId, popart::IArray &> anchors = {
        {out, outData},
    };

    session->prepareDevice();

    popart::NDArrayWrapper<float> inData(vInData.data(), inShape);
    std::map<popart::TensorId, popart::IArray &> inputs = {{inId, inData}};

    popart::StepIO stepio(inputs, anchors);

    session->run(stepio);

    auto t1 = steady_clock::now();
    auto dX = duration<double, std::milli>(t1 - t0).count();
    std::cout << "seed=" << seed << "  N=" << N << "  J=" << J
              << "  inplace=" << inp << "  outline=" << outlining
              << "  perturbInput=" << perturbInput << "  time=" << dX << " [ms]"
              << std::endl;

    return rawOutputData;
  };

  auto runTest = [&getValue, perturbSize](int seed, int N, int J) {
    // vanilla+outline
    auto v010 = getValue(false, seed, N, J, true, false);
    // vanilla
    auto v000 = getValue(false, seed, N, J, false, false);
    // vanilla+inplace
    auto v100 = getValue(true, seed, N, J, false, false);
    // vanilla+inplace+outline
    auto v110 = getValue(true, seed, N, J, true, false);
    // vanilla+perturb
    auto v001 = getValue(false, seed, N, J, false, true);

    std::cout << std::scientific << "Final value with Vanilla is " << v000
              << ". Some discrepencies : "
              << "\nInplace to Vanilla : " << std::fabs(v100 - v000)
              << "\nOutline to Vanilla : " << std::fabs(v010 - v000)
              << "\nPerturb to Vanilla : " << std::fabs(v001 - v000)
              << "\nInplace to Outline : " << std::fabs(v110 - v010)
              << "\nInplace and outline to Vanilla : " << std::fabs(v110 - v000)
              << std::endl;

    // hardware addition is not commutative, we cannot expect bit-wise
    // agreement.

    // inplacing, no outlining:
    BOOST_CHECK(std::fabs(v100 - v000) < 1e-4);
    // inplacing and outlining:
    BOOST_CHECK(std::fabs(v110 - v000) < 1e-4);
    // this is not a useless test, permute input has an effect.
    BOOST_CHECK(std::fabs(v001 - v000) > 1e-2 * perturbSize);
  };

  int seed = 1013;
  int J    = 3;
  int N    = 3;
  runTest(seed, N, J);
}
