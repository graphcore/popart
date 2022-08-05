// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE PipelineTrainingTest0

#include <algorithm>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/test/unit_test.hpp>
#include <cstdint>
#include <cstdlib>
#include <filereader.hpp>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <testdevice.hpp>
#include <tuple>
#include <utility>
#include <vector>
#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/inputshapeinfo.hpp>
#include <popart/ndarraywrapper.hpp>
#include <popart/op/ipucopy.hpp>
#include <popart/session.hpp>
#include <popart/sgd.hpp>
#include <popart/tensorinfo.hpp>

#include "../random_util.hpp"
#include "popart/builder.gen.hpp"
#include "popart/ir.hpp"
#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/patterns/patterns.hpp"
#include "popart/sessionoptions.hpp"
#include "popart/stepio.hpp"
#include "popart/tensordebuginfo.hpp"
#include "popart/voiddata.hpp"

namespace popart {
class IArray;
} // namespace popart

// In this model, where continuous and exact pipelines are numerically
// equivalent, there are Ops in the forwards and backwards passes which are
// discontiguous. We test that the Ir transformation of inserting IPUCopys are
// correct, as well as that the numerical output agrees between the exact and
// continuous cases.
BOOST_AUTO_TEST_CASE(DiscontiguousIpuCopyTest0) {

  using namespace popart;

  bool printStdOut = true;

  enum class TestType { Numerical, Ir };

  auto test = [printStdOut](TestType tt) {
    // input stream samples are generated randomly
    int seed = 1011;
    DefaultRandomEngine eng(seed);
    UniformRealDistribution<float> fdis(0.f, 1.f);

    int64_t batchSize      = 4;
    int64_t batchesPerStep = 400;
    int64_t sampleHeight   = 3;
    int64_t samplesPerStep = batchesPerStep * batchSize;
    std::vector<int64_t> sampleShape{sampleHeight, sampleHeight};
    std::vector<int64_t> weightShape = sampleShape;
    std::vector<int64_t> batchShape{batchSize, sampleHeight, sampleHeight};
    std::vector<int64_t> stepDataShape{
        batchesPerStep, batchSize, sampleHeight, sampleHeight};
    TensorInfo sampleInfo{"FLOAT", sampleShape};
    TensorInfo weightInfo = sampleInfo;
    TensorInfo batchInfo{"FLOAT", batchShape};
    TensorInfo stepDataInfo{"FLOAT", stepDataShape};
    int64_t sampleElms{sampleHeight * sampleHeight};
    int64_t batchElms    = sampleElms * batchSize;
    int64_t stepDataElms = batchElms * batchesPerStep;

    // The model:
    //
    //  input1              input2
    //    |                   |
    //   (Add) -- Weight     (Add) -- Weight
    //    |                   |
    //   (Add) -- Weight     (Add) -- Weight
    //    |                   |
    //   (Add) -- Weight     (Add) -- Weight
    //    |                   |
    //   (Add) -- Weight     (Add) -- Weight
    //    |                   |
    //   (Add) -- Weight     (Add) -- Weight
    //    \                   |
    //     \                  |
    //      \----------(Add)--|
    //                   |
    //                finalOut
    //                   |
    //                 l1-loss
    //
    // Having two branches like this ensures that there is a discontiguous
    // IPUCopy (from one the 2 branches to the IPU where the loss is computed)

    // number of Adds on each of the two branches.
    int nLayers = 10;

    auto builder     = Builder::create();
    auto aiOnnx      = builder->aiOnnxOpset9();
    auto aiGraphcore = builder->aiGraphcoreOpset1();
    auto input0      = builder->addInputTensor(batchInfo, "0tupni");
    auto input1      = builder->addInputTensor(batchInfo, "1tupni");

    // Storage for all layers
    std::vector<std::vector<float>> allWeights;
    std::vector<ConstVoidData> allWeightCvds;
    std::vector<TensorId> allWeightIds;
    std::vector<TensorId> allActIds;
    std::vector<std::vector<float>> w_readbacks;
    WeightsIO weightsRead;

    int nLayersAdded    = 0;
    auto insertAddLayer = [&allWeights,
                           &allWeightCvds,
                           &allWeightIds,
                           &allActIds,
                           sampleElms,
                           &weightsRead,
                           weightInfo,
                           sampleInfo,
                           &builder,
                           &aiOnnx,
                           &nLayersAdded,
                           &w_readbacks](TensorId actInId) {
      w_readbacks.push_back(std::vector<float>(sampleElms, -99.0f));
      allWeights.push_back(std::vector<float>(sampleElms, 0.0f));
      allWeightCvds.push_back({allWeights.back().data(), sampleInfo});
      allWeightIds.push_back(
          builder->addInitializedInputTensor(allWeightCvds.back()));
      TensorId actOutId = "act" + std::to_string(nLayersAdded);
      allActIds.push_back(aiOnnx.add({allWeightIds.back(), actInId}, actOutId));
      weightsRead.insert(allWeightIds.back(),
                         {w_readbacks.back().data(), weightInfo});
      ++nLayersAdded;
    };

    // left branch (branch 0)
    insertAddLayer(input0);
    for (int i = 1; i < nLayers; ++i) {
      insertAddLayer(allActIds.back());
    }
    TensorId actFinal0 = allActIds.back();

    // right branch (branch 1)
    insertAddLayer(input1);
    for (int i = 1; i < nLayers; ++i) {
      insertAddLayer(allActIds.back());
    }
    TensorId actFinal1 = allActIds.back();

    // sum of the 2 branch outputs
    auto actFinal = aiOnnx.add({actFinal0, actFinal1}, "finalAct");

    float learnRate = 0.01;
    auto optimizer  = ConstSGD(learnRate);

    float lambda = 0.1;
    actFinal     = builder->aiGraphcoreOpset1().l1loss(
        {actFinal}, lambda, ReductionType::Sum);

    auto proto = builder->getModelProto();
    // No anchors
    auto dataFlow = DataFlow(batchesPerStep);

    auto device = createTestDevice(TEST_TARGET, 7);

    SessionOptions userOptions;
    userOptions.virtualGraphMode = VirtualGraphMode::Auto;
    userOptions.enablePipelining = true;

    if (tt == TestType::Ir) {

      std::vector<std::tuple<int64_t, int64_t>> pipeSrcDsts;

      auto getIpuCopies = [](const Ir &ir) {
        std::vector<IpuCopyOp *> copies;
        for (const auto &x : ir.getMainGraphOps()) {
          auto ipuCopy = dynamic_cast<IpuCopyOp *>(x.second.get());
          if (ipuCopy) {
            copies.push_back(ipuCopy);
          }
        }
        return copies;
      };

      auto modelProto = io::getModelFromString(proto);

      Ir irWithPipe;
      irWithPipe.prepare({modelProto,
                          InputShapeInfo(),
                          dataFlow,
                          actFinal,
                          &optimizer,
                          *device,
                          userOptions,
                          Patterns(PatternsLevel::Default)});

      auto copiesWithPipe = getIpuCopies(irWithPipe);
      for (auto cop : copiesWithPipe) {
        pipeSrcDsts.push_back(
            std::make_tuple(cop->getSourceIpu(), cop->getDestIpu()));
      }

      userOptions.enablePipelining = false;
      Ir irWithoutPipe;
      irWithoutPipe.prepare({modelProto,
                             InputShapeInfo(),
                             dataFlow,
                             actFinal,
                             &optimizer,
                             *device,
                             userOptions,
                             Patterns(PatternsLevel::Default)});

      // we are testing both discontiguous copies in the forward and backward
      // passes. So  check that we actually have both types in the graph.
      bool fwdDisco = false;
      bool bwdDisco = false;

      auto copiesWithoutPipe = getIpuCopies(irWithoutPipe);
      // we compute on-the-fly what the expected copies are
      std::map<std::string, std::tuple<int64_t, int64_t>> expectedSrcDstsMap;
      for (auto cop : copiesWithoutPipe) {
        auto ipuDiff = static_cast<int64_t>(cop->getDestIpu()) -
                       static_cast<int64_t>(cop->getSourceIpu());
        if (ipuDiff < -1) {
          bwdDisco = true;
        }
        if (ipuDiff > +1) {
          fwdDisco = true;
        }

        auto delta = cop->getDestIpu() < cop->getSourceIpu() ? -1 : +1;
        auto inId  = cop->inId(0);
        for (auto src = cop->getSourceIpu(); src != cop->getDestIpu();
             src += delta) {
          std::string id         = inId + "__from__" + std::to_string(src);
          expectedSrcDstsMap[id] = std::make_tuple(src, src + delta);
        }
      }

      std::vector<std::tuple<int64_t, int64_t>> expectedSrcDsts;
      for (auto &id_srcDst : expectedSrcDstsMap) {
        expectedSrcDsts.push_back(id_srcDst.second);
      }

      BOOST_CHECK(fwdDisco == true);
      BOOST_CHECK(bwdDisco == true);

      std::sort(pipeSrcDsts.begin(), pipeSrcDsts.end());
      std::sort(expectedSrcDsts.begin(), expectedSrcDsts.end());

      if (printStdOut) {
        std::cout << "With pipelining: " << std::endl;
        for (auto ipuCopy : copiesWithPipe) {
          std::cout << ipuCopy->getFromToStr() << std::endl;
        }
        std::cout << "----------------" << std::endl;
        std::cout << "Without pipelining: " << std::endl;
        for (auto ipuCopy : copiesWithoutPipe) {
          std::cout << ipuCopy->getFromToStr() << std::endl;
        }
        std::cout << "----------------" << std::endl;
        std::cout << "PipeSrcDsts: " << std::endl;
        for (auto &srcDst : pipeSrcDsts) {
          auto src = std::get<0>(srcDst);
          auto dst = std::get<1>(srcDst);
          std::cout << "[ " << src << " ] --> [ " << dst << " ]" << std::endl;
        }
        std::cout << "----------------" << std::endl;
        std::cout << "ExpectedSrcDsts: " << std::endl;
        for (auto &srcDst : expectedSrcDsts) {
          auto src = std::get<0>(srcDst);
          auto dst = std::get<1>(srcDst);
          std::cout << "[ " << src << " ] --> [ " << dst << " ]" << std::endl;
        }
      }

      BOOST_CHECK(pipeSrcDsts == expectedSrcDsts);
    }

    // numerical test: compare computed weights after several iterations,
    // compare to the expected weights (which are easy to compute as the model
    // is linear)
    else if (tt == TestType::Numerical) {

      auto session = popart::TrainingSession::createFromOnnxModel(
          proto,
          dataFlow,
          actFinal,
          optimizer,
          device,
          InputShapeInfo(),
          userOptions,
          popart::Patterns(PatternsLevel::Default));

      session->prepareDevice();

      // The samples (same for 0 and 1)
      std::vector<float> v_input_x(stepDataElms);

      // cumulative samples (accumulated over multiple steps).
      std::vector<float> v_sample_sum_x(weightInfo.nelms(), 0.0f);
      std::map<popart::TensorId, popart::IArray &> anchors = {};

      // write initial weights to host
      session->weightsFromHost();

      float sampleNumVal = 100.0f;
      for (int i = 0; i < 3; ++i) {
        std::cout << "Iteration (call to run(...)) # " << i << std::endl;

        // generate new samples
        for (int i = 0; i < samplesPerStep; ++i) {
          for (int j = 0; j < sampleElms; ++j) {
            auto stepIndex = i * sampleElms + j;
            v_input_x[stepIndex] =
                fdis(eng) > 0.5 ? -sampleNumVal : +sampleNumVal;
            v_sample_sum_x[j] += v_input_x[stepIndex];
          }
        }
        popart::NDArrayWrapper<float> input_x_wrapper(v_input_x.data(),
                                                      stepDataInfo);
        std::map<popart::TensorId, popart::IArray &> inputs = {
            {input1, input_x_wrapper}, {input0, input_x_wrapper}};
        popart::StepIO stepio(inputs, anchors);

        // process the samples
        session->run(stepio);
      }

      // read final weights back
      session->weightsToHost();
      session->readWeights(weightsRead);

      // get sum of absolute differences between computed and expected
      float sumAbsDiff = 0.0;
      for (auto &wv : w_readbacks) {
        for (int i = 0; i < wv.size(); ++i) {

          if (printStdOut) {
            std::cout << "Returned : " << wv[i]
                      << "   - learnRate * lambda * sum / sampleNumVal : "
                      << -v_sample_sum_x[i] * learnRate * lambda / sampleNumVal
                      << std::endl;
          }
          sumAbsDiff += std::abs(wv[i] + v_sample_sum_x[i] * learnRate *
                                             lambda / sampleNumVal);
        }
      }
      BOOST_CHECK(sumAbsDiff < 1e-5);
    }
  };

  test(TestType::Ir);
  test(TestType::Numerical);
}
