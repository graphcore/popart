// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE MergeMultiSgdVarUpdatesTransformation0

#include <../../random_util.hpp>
#include <boost/test/unit_test.hpp>
#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/devicemanager.hpp>
#include <popart/filereader.hpp>
#include <popart/inputshapeinfo.hpp>
#include <popart/ndarraywrapper.hpp>
#include <popart/op/identity.hpp>
#include <popart/op/l1.hpp>
#include <popart/optimizer.hpp>
#include <popart/session.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/tensornames.hpp>
#include <popart/testdevice.hpp>

#include <algorithm>
#include <map>
#include <memory>
#include <tuple>
#include <vector>

using namespace popart;

namespace {
TensorId
bnconv(Builder *b, TensorId act, ConstVoidData cwdata, ConstVoidData bwdata) {
  auto aiOnnx      = b->aiOnnxOpset9();
  auto convWeights = b->addInitializedInputTensor(cwdata);
  auto scale       = b->addInitializedInputTensor(bwdata);
  auto bias        = b->addInitializedInputTensor(bwdata);
  auto mean        = b->addInitializedInputTensor(bwdata);
  auto var         = b->addInitializedInputTensor(bwdata);
  auto act0 =
      aiOnnx.conv({act, convWeights}, {1, 1}, 1, {}, {1, 1, 1, 1}, {1, 1});
  auto act1 = aiOnnx.batchnormalization({act0, scale, bias, mean, var}, 5);
  // return the output of batch-norm
  return act1[0];
}
} // namespace

BOOST_AUTO_TEST_CASE(Transformation_MergeMultiSGD) {

  auto test = [](MergeVarUpdateType mvu) {
    // we will generate random input data
    int seed = 1013;
    DefaultRandomEngine eng(seed);
    UniformRealDistribution<float> fdis(-4.f, +4.f);

    // construct onnx model
    auto builder = Builder::create();
    auto aiOnnx  = builder->aiOnnxOpset9();

    int nInChans  = 3;
    int batchsize = 1;
    TensorInfo in0info{"FLOAT",
                       std::vector<int64_t>{batchsize, nInChans, 6, 6}};
    auto in0 = builder->addInputTensor(in0info);
    std::vector<float> in0data(in0info.nelms());
    for (auto &val : in0data) {
      val = fdis(eng);
    }

    // The model will be
    // reduce(bn(conv(bn(conv(bn(conv(bn(conv(....(bn(conv(input))...)))), nConv
    // convolutions chained together, with the number of channels increasing by
    // 1 at each subsequent conv.
    constexpr int nConv = 8;
    std::array<std::vector<float>, nConv> convWeights;
    std::array<std::vector<float>, nConv> bnWeights;
    std::array<ConstVoidData, nConv> cvds;
    std::array<ConstVoidData, nConv> bnds;
    std::array<TensorId, nConv + 1> actIds;
    actIds[0] = in0;

    int64_t copyElms = 0;
    int64_t sgdElms  = 0;
    for (int i = 0; i < nConv; ++i) {
      int nOutChans = nInChans + 1;
      TensorInfo conv_weight_info{
          "FLOAT", std::vector<int64_t>{nOutChans, nInChans, 1, 1}};
      TensorInfo bn_weight_info("FLOAT", std::vector<int64_t>{nOutChans});

      copyElms += 2 * nOutChans;
      sgdElms += 2 * nOutChans;
      sgdElms += nOutChans * nInChans;

      convWeights[i] = std::vector<float>(nOutChans * nInChans, 0);
      for (auto &x : convWeights[i]) {
        x = fdis(eng);
      }
      bnWeights[i] = std::vector<float>(nOutChans, 0);
      for (auto &x : bnWeights[i]) {
        x = fdis(eng);
      }

      cvds[i] = {convWeights[i].data(), conv_weight_info};
      bnds[i] = {bnWeights[i].data(), bn_weight_info};

      actIds[i + 1] = bnconv(builder.get(), actIds[i], cvds[i], bnds[i]);
      nInChans      = nOutChans;
    }

    auto reduced     = aiOnnx.reducesum({actIds[nConv]}, {1, 2, 3});
    float lossLambda = 0.26;
    auto l1 = builder->aiGraphcoreOpset1().l1loss({reduced}, lossLambda);
    std::string proto = builder->getModelProto();
    auto modelProto   = io::getModelFromString(proto);

    // create the IR
    auto art      = AnchorReturnType("All");
    auto dataFlow = DataFlow(1, {{reduced, art}});

    auto device = popart::createTestDevice(TEST_TARGET);

    auto opts = SessionOptions();

    // accurate scheduling is quite slow for this one. Imposing this limit on
    // how many swaps to perform in the scheduling algorithm reduces test time
    // from 280 seconds to 50 seconds.
    opts.swapLimitScheduler = 20;
    opts.firstDotOp         = 0;
    opts.finalDotOp         = 200;
    opts.dotChecks.insert(DotCheck::Final);
    opts.logDir                     = ".";
    opts.mergeVarUpdate             = mvu;
    opts.mergeVarUpdateMemThreshold = mvu == MergeVarUpdateType::AutoLoose
                                          ? 10000
                                          : 24; // 24 bytes = 7 floats
    opts.looseThresholdAtPeak = 10000;

    float learningRate = 0.1;
    auto optimizer     = SGD({{"defaultLearningRate", {learningRate, false}}});

    Ir ir;
    ir.prepare({modelProto,
                InputShapeInfo(),
                dataFlow,
                l1,
                &optimizer,
                *device,
                opts,
                Patterns()});

    // Check the ir
    //
    // For case MergeVarUpdateType::All, all the ConstSgdVarUpdates have the
    // same learning rate, weight decay, so should all be merged into a single
    // group

    // Lambda to check the number of ops are as expected.
    auto checkOps = [&](const popart::AiGraphcoreOpIdV1 &opType, int expected) {
      auto count = ir.opsOfType(opType).size();
      std::cout << opType.type << " Elements : " << count
                << " Expected: " << expected << std::endl;
      BOOST_CHECK(count == expected);
    };
    if (mvu == MergeVarUpdateType::All) {
      // 10 flattens per layer are:
      // 3) conv filters, bn bias, bn scale
      // 3) grads of each of the above
      // 2) running mean & variance
      // 2) updates for each of the above

      checkOps(Onnx::CustomOperators::FlattenInplace, nConv * 10);

      // 4 ConcatInplace
      checkOps(Onnx::CustomOperators::ConcatInplace, 4);
      checkOps(Onnx::CustomOperators::SGD0VarUpdate, 1);
      checkOps(Onnx::CustomOperators::CopyVarUpdate, 1);

    } else if (mvu == MergeVarUpdateType::None) {
      checkOps(Onnx::CustomOperators::SGD0VarUpdate, 3 * nConv);
      checkOps(Onnx::CustomOperators::CopyVarUpdate, 2 * nConv);
      checkOps(Onnx::CustomOperators::FlattenInplace, 0);
      checkOps(Onnx::CustomOperators::ConcatInplace, 0);
    }

    else if (mvu == MergeVarUpdateType::AutoTight) {
      auto thr = opts.mergeVarUpdateMemThreshold;
      std::cout << "memory threshold : " << thr << std::endl;
      auto expectednsgd = (sgdElms * 4) / thr + ((sgdElms * 4) % thr != 0);
      checkOps(Onnx::CustomOperators::SGD0VarUpdate, expectednsgd);

      auto expectedncopy = (copyElms * 4) / thr + ((copyElms * 4) % thr != 0);
      checkOps(Onnx::CustomOperators::CopyVarUpdate, expectedncopy);
    }

    else if (mvu == MergeVarUpdateType::AutoLoose) {
      // because both thresholds are greater than the total memory, there should
      // be just 1 SGDVarUpdate and 1 CopyVarUpdate
      checkOps(Onnx::CustomOperators::SGD0VarUpdate, 1);
      checkOps(Onnx::CustomOperators::CopyVarUpdate, 1);
    }
  };

  test(MergeVarUpdateType::All);
  test(MergeVarUpdateType::None);
  test(MergeVarUpdateType::AutoTight);
  test(MergeVarUpdateType::AutoLoose);
}
