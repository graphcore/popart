#define BOOST_TEST_MODULE NumericsInplaceVsNot0Test

#include <boost/filesystem.hpp>
#include <boost/test/unit_test.hpp>
#include <poponnx/builder.hpp>
#include <poponnx/dataflow.hpp>
#include <poponnx/devicemanager.hpp>
#include <poponnx/filereader.hpp>
#include <poponnx/inputshapeinfo.hpp>
#include <poponnx/ir.hpp>
#include <poponnx/names.hpp>
#include <poponnx/ndarraywrapper.hpp>
#include <poponnx/op/l1.hpp>
#include <poponnx/optimizer.hpp>
#include <poponnx/session.hpp>
#include <poponnx/tensordata.hpp>

#include <iostream>
#include <random>

using namespace poponnx;

// Module:      |  .           -|
//              |- slice [1,2] -|
//              |- slice [1,2] -|-- top half: i :slice 2*i + slice 2*i+1 -|
//              |  .            |                                         /
// in [2^N,2] --|  .            |                                        /
//              |  .            |                                       /
//              |  .            |=-=-=-=-                              /-
//              |  .            |                                     /  |
//              |  .            |                                    /   |
//              |  .            |-- bottom half: scale random slice /    |
//              |- slice [1,2] -|                                       /
//              |- slice [1,2] -|                                      /
//              |  .           -|                                     /
//              |  .           -|                         contig concats into
//                                                        3 regions of sizes
//                                                           (2^N)/4+1
//                                                          (2^N)/4
//                                                         (2^N)/2-1.
//                                                            /
//                                                   concat the above 3
//                                                       [2,0,1]
//                                                         /
//                                                        /
//                                                       /
//                                               output is [2^N,2]
//
// The Module is repeated J times, then there is a sum reduce:
//
//                    J repeats of the Module
//                          |
//                   ------------------
//                   |          |      |
//  in [2^N,2] -- Module -- Module -- Module -- ReduceSum

BOOST_AUTO_TEST_CASE(Inplace_numericsIpNip0) {
  auto getValue = [](bool inp, int seed) {
    int N = 2;
    int J = 5;

    // The input is of shape (4^N, 2)
    Shape inShape{static_cast<int64_t>(std::pow(2, N)), 2};
    TensorInfo inInfo{"FLOAT", inShape};

    // generate random input data
    std::default_random_engine eng(seed);
    std::uniform_real_distribution<float> fdis(-4, 4);
    std::uniform_int_distribution<uint64_t> idis(
        0, std::numeric_limits<uint64_t>::max());

    std::vector<float> vInData(inInfo.nelms());
    for (auto &val : vInData) {
      val = fdis(eng);
    }

    // Build an onnx model
    auto builder = Builder::create();
    auto aiOnnx  = builder->aiOnnxOpset9();

    auto appendModule = [N, &builder, &eng, &fdis, &aiOnnx, &idis](
                            std::vector<TensorId> tensorsIn) {
      std::vector<TensorId> tensorsOut;
      // the top half:
      for (int i = 0; i < std::pow(2, N - 1); ++i) {
        auto sum        = aiOnnx.add({tensorsIn[2 * i], tensorsIn[2 * i + 1]});
        auto sigmoidOut = aiOnnx.sigmoid({sum});
        tensorsOut.push_back(sigmoidOut);
      }

      // the bottom half:
      for (int i = 0; i < std::pow(2, N - 1); ++i) {
        TensorId concatId0 = tensorsIn[idis(eng) % tensorsIn.size()];
        TensorId concatId1 = tensorsIn[idis(eng) % tensorsIn.size()];
        auto concatOut     = aiOnnx.concat({concatId0, concatId1}, 0);
        builder->setInplacePreferences(concatOut,
                                       {{"ConcatInplace", 100.0f + fdis(eng)}});
        auto scaleOut = aiOnnx.scale({concatOut}, fdis(eng));
        builder->setInplacePreferences(scaleOut,
                                       {{"ScaleInplace", 100.0f + fdis(eng)}});
        tensorsOut.push_back(scaleOut);
      }
      int start0 = 0;
      int start1 = std::pow(2, N - 2) + 1;
      int start2 = start1 + std::pow(2, N - 2);

      std::vector<TensorId> tensorIds0{tensorsOut.begin() + start0,
                                       tensorsOut.begin() + start1};
      std::vector<TensorId> tensorIds1{tensorsOut.begin() + start1,
                                       tensorsOut.begin() + start2};
      std::vector<TensorId> tensorIds2{tensorsOut.begin() + start2,
                                       tensorsOut.begin() + std::pow(2, N)};

      auto concat0 = aiOnnx.concat(tensorIds0, 0);
      builder->setInplacePreferences(concat0,
                                     {{"ConcatInplace", 100.0f + fdis(eng)}});
      auto concat1 = aiOnnx.concat(tensorIds1, 0);
      builder->setInplacePreferences(concat1,
                                     {{"ConcatInplace", 100.0f + fdis(eng)}});
      auto concat2 = aiOnnx.concat(tensorIds2, 0);
      builder->setInplacePreferences(concat2,
                                     {{"ConcatInplace", 100.0f + fdis(eng)}});

      auto outcon = aiOnnx.concat({concat2, concat0, concat1}, 0);
      builder->setInplacePreferences(outcon,
                                     {{"ConcatInplace", 100.0f + fdis(eng)}});

      return outcon;
    };

    auto getSliced = [N, &builder, &aiOnnx, &fdis, &eng](TensorId in) {
      std::vector<TensorId> slicedIds;
      for (int i = 0; i < std::pow(2, N); ++i) {
        auto sliceOut = aiOnnx.slice({in}, {i + 1, 2}, {i, 0}, {0, 1});
        builder->setInplacePreferences(sliceOut,
                                       {{"SliceInplace", 100.0f + fdis(eng)}});
        slicedIds.push_back(sliceOut);
      }
      return slicedIds;
    };

    auto inId = builder->addInputTensor(inInfo);

    auto singleTensor = aiOnnx.sigmoid({inId});
    for (int i = 0; i < J; ++i) {
      // break the input into many [1,2] tensors:
      auto slicedTensors = getSliced(singleTensor);
      singleTensor       = appendModule(slicedTensors);
    }

    auto out = aiOnnx.reducesum({singleTensor}, {0, 1}, false);
    builder->addOutputTensor(out);

    auto proto      = builder->getModelProto();
    auto modelProto = io::getModelFromString(proto);

    // Create the IR, adding outId as an anchor
    auto art      = AnchorReturnType("ALL");
    auto dataFlow = DataFlow(1, {{out, art}});

    auto opts = SessionOptions();
    // just the one .dot file will be written
    opts.dotChecks.insert(DotCheck::BWD0);
    opts.dotChecks.insert(DotCheck::FWD0);
    opts.dotChecks.insert(DotCheck::FINAL);
    opts.dotOpNames = false;
    opts.logDir     = "./dotfiles";
    boost::filesystem::create_directory(opts.logDir);

    auto session = poponnx::InferenceSession::createFromOnnxModel(
        proto,
        dataFlow,
        {},
        poponnx::InputShapeInfo(),
        opts,
        poponnx::Patterns(PatternsLevel::NONE).enableInPlace(inp));

    auto cpuDevice =
        poponnx::DeviceManager::createDeviceManager().createCpuDevice();
    session->setDevice(*cpuDevice);

    // prepare the anchors
    float rawOutputData;
    Shape outShape{};
    poponnx::NDArrayWrapper<float> outData(&rawOutputData, outShape);

    std::map<poponnx::TensorId, poponnx::IArray &> anchors = {
        {out, outData},
    };

    session->prepareDevice();

    poponnx::NDArrayWrapper<float> inData(vInData.data(), inShape);
    std::map<poponnx::TensorId, poponnx::IArray &> inputs = {{inId, inData}};

    poponnx::StepIO stepio(inputs, anchors);

    session->run(stepio);

    return rawOutputData;
  };

  auto vFalse  = getValue(false, 1013);
  auto vTrue   = getValue(true, 1013);
  auto absDiff = std::abs<float>(vFalse - vTrue);

  logging::ir::debug(
      "Network output wih Inplace ON : {} and OFF {}, absDiff {}",
      vTrue,
      vFalse,
      absDiff);

  BOOST_CHECK(absDiff < 1e-6);
}
