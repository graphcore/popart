#define BOOST_TEST_MODULE MergeMultiSgdVarUpdatesTransformation0

#include <boost/test/unit_test.hpp>
#include <poponnx/builder.hpp>
#include <poponnx/dataflow.hpp>
#include <poponnx/devicemanager.hpp>
#include <poponnx/filereader.hpp>
#include <poponnx/inputshapeinfo.hpp>
#include <poponnx/ndarraywrapper.hpp>
#include <poponnx/op/l1.hpp>
#include <poponnx/optimizer.hpp>
#include <poponnx/session.hpp>
#include <poponnx/tensorinfo.hpp>
#include <poponnx/tensornames.hpp>

#include <algorithm>
#include <map>
#include <random>
#include <tuple>
#include <vector>

using namespace poponnx;

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
    std::default_random_engine eng(seed);
    std::uniform_real_distribution<float> fdis(-4, 4);

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

    auto reduced      = aiOnnx.reducesum({actIds[nConv]}, {1, 2, 3});
    std::string proto = builder->getModelProto();
    auto modelProto   = io::getModelFromString(proto);

    // create the IR
    auto art      = AnchorReturnType("ALL");
    auto dataFlow = DataFlow(1, {{reduced, art}});

    auto cpuDevice =
        poponnx::DeviceManager::createDeviceManager().createCpuDevice();

    auto opts = SessionOptions();

    opts.firstDotOp = 0;
    opts.finalDotOp = 200;
    opts.dotChecks.insert(DotCheck::FINAL);
    opts.logDir                     = ".";
    opts.mergeVarUpdate             = mvu;
    opts.mergeVarUpdateMemThreshold = 24; // 24 bytes = 7 floats

    float lossLambda = 0.26;
    float learnRate  = 0.1;
    auto optimizer   = SGD(learnRate);
    std::vector<Loss *> losses{
        new L1Loss(reduced, "l1LossVal", lossLambda, ReductionType::SUM)};

    Ir ir;
    ir.prepare({modelProto,
                InputShapeInfo(),
                dataFlow,
                losses,
                &optimizer,
                *cpuDevice,
                opts,
                Patterns()});

    // Check the ir
    //
    // For case MergeVarUpdateType::All, all the ConstSgdVarUpdates have the
    // same learning rate, weight decay, so should all be merged into a single
    // group
    if (mvu == MergeVarUpdateType::All) {
      // 10 flattens per layer are:
      // 3) conv filters, bn bias, bn scale
      // 3) grads of each of the above
      // 2) running mean & variance
      // 2) updates for each of the above

      BOOST_CHECK(ir.opsOfType(Onnx::CustomOperators::FlattenInplace).size() ==
                  nConv * 10);

      // check that no ConstSgdVarUpdate entered
      BOOST_CHECK(
          ir.opsOfType(Onnx::CustomOperators::ConstSgdVarUpdate).size() == 0);

      // 4 ConcatInplace
      BOOST_CHECK(ir.opsOfType(Onnx::CustomOperators::ConcatInplace).size() ==
                  4);

      BOOST_CHECK(ir.opsOfType(Onnx::CustomOperators::SgdVarUpdate).size() ==
                  1);
      BOOST_CHECK(ir.opsOfType(Onnx::CustomOperators::CopyVarUpdate).size() ==
                  1);

    } else if (mvu == MergeVarUpdateType::None) {
      BOOST_CHECK(ir.opsOfType(Onnx::CustomOperators::SgdVarUpdate).size() ==
                  3 * nConv);
      BOOST_CHECK(ir.opsOfType(Onnx::CustomOperators::CopyVarUpdate).size() ==
                  2 * nConv);
      BOOST_CHECK(ir.opsOfType(Onnx::CustomOperators::FlattenInplace).size() ==
                  0);
      BOOST_CHECK(ir.opsOfType(Onnx::CustomOperators::ConcatInplace).size() ==
                  0);
    }

    else if (mvu == MergeVarUpdateType::AutoTight) {
      auto nSgd  = ir.opsOfType(Onnx::CustomOperators::SgdVarUpdate).size();
      auto nCopy = ir.opsOfType(Onnx::CustomOperators::CopyVarUpdate).size();
      auto thr   = opts.mergeVarUpdateMemThreshold;
      std::cout << "copy elms : " << copyElms << std::endl;
      std::cout << "sgd elms : " << sgdElms << std::endl;
      std::cout << "nSgd : " << nSgd << std::endl;
      std::cout << "nCopy : " << nCopy << std::endl;
      std::cout << "memory threshold : " << thr << std::endl;
      auto expectednsgd = (sgdElms * 4) / thr + ((sgdElms * 4) % thr != 0);
      std::cout << "expected sgds : " << expectednsgd << std::endl;
      auto expectedncopy = (copyElms * 4) / thr + ((copyElms * 4) % thr != 0);
      std::cout << "expected copies : " << expectedncopy << std::endl;
      BOOST_CHECK(nSgd == expectednsgd);
      BOOST_CHECK(nCopy == expectedncopy);
    }
  };

  test(MergeVarUpdateType::All);
  test(MergeVarUpdateType::None);
  test(MergeVarUpdateType::AutoTight);
}
