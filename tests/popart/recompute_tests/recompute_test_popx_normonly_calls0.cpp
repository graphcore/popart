#define BOOST_TEST_MODULE RecomputeTestPopxNormOnlyCalls0

#include <boost/test/unit_test.hpp>
#include <iostream>
#include <vector>

#include <popart/builder.hpp>
#include <popart/filereader.hpp>
#include <popart/op.hpp>
#include <popart/op/l1.hpp>
#include <popart/optimizer.hpp>

#define private public
#define protected public
#include <popart/popx/devicex.hpp>
#include <popart/session.hpp>
#undef private
#undef public

// TODO(jn) check values correct in python tests

using namespace popart;

TensorId conv(Builder *b, TensorId act, ConstVoidData wdata) {
  auto aiOnnx  = b->aiOnnxOpset9();
  auto weights = b->addInitializedInputTensor(wdata);
  act = aiOnnx.conv({act, weights}, {1, 1}, 1, {}, {0, 0, 0, 0}, {1, 1});
  return act;
}

TensorId batchnormalization(Builder *b, TensorId act, ConstVoidData bndata) {
  auto aiOnnx = b->aiOnnxOpset9();
  auto scale  = b->addInitializedInputTensor(bndata);
  auto bias   = b->addInitializedInputTensor(bndata);
  auto mean   = b->addInitializedInputTensor(bndata);
  auto var    = b->addInitializedInputTensor(bndata);
  auto bn_out = aiOnnx.batchnormalization({act, scale, bias, mean, var}, 5);
  act         = bn_out.at(0);
  return act;
}

BOOST_AUTO_TEST_CASE(RecomputeTestPopxNormOnlyCalls0) {

  auto builder = Builder::create();
  auto aiOnnx  = builder->aiOnnxOpset9();
  TensorInfo input_info{"FLOAT", std::vector<int64_t>{1, 4, 32, 32}};

  // convolution,  4 channels -> 4 channels
  TensorInfo weights_info{"FLOAT", std::vector<int64_t>{4, 4, 1, 1}};
  float weight_vals[4 * 4 * 1 * 1] = {0};
  ConstVoidData weight_data        = {weight_vals, weights_info};

  // batch-normalization
  TensorInfo bn_info{"FLOAT", std::vector<int64_t>{4}};
  float bn_vals[4]      = {0};
  ConstVoidData bn_data = {bn_vals, bn_info};

  int nLayers = 8;
  auto act    = builder->addInputTensor(input_info);
  for (int i = 0; i < nLayers; ++i) {
    act = conv(builder.get(), act, weight_data);
    act = batchnormalization(builder.get(), act, bn_data);
    act = aiOnnx.relu({act});
  }
  builder->addOutputTensor(act);

  auto proto      = builder->getModelProto();
  auto modelProto = io::getModelFromString(proto);

  // Create the IR
  auto art       = AnchorReturnType("ALL");
  auto dataFlow  = DataFlow(1, {{act, art}});
  auto optimizer = ConstSGD(0.01);

  auto l1loss = std::unique_ptr<L1Loss>(
      new L1Loss(act, "l1LossVal", 0.1, ReductionType::SUM));
  std::vector<Loss *> losses{l1loss.get()};

  auto cpuDevice =
      popart::DeviceManager::createDeviceManager().createCpuDevice();

  auto opts              = SessionOptions();
  opts.autoRecomputation = RecomputationType::NormOnly;
  opts.enableOutlining   = false;

  auto session = popart::TrainingSession::createFromOnnxModel(
      proto,
      dataFlow,
      losses,
      optimizer,
      cpuDevice,
      InputShapeInfo(),
      opts,
      Patterns({popart::PreAliasPatternType::POSTNREPL,
                popart::PreAliasPatternType::CONVDATAGRAD}));
  session->prepareDevice();

  popart::popx::Devicex *devicex =
      dynamic_cast<popart::popx::Devicex *>(session->device_.get());

  // we count how many times each op appears
  std::map<Op *, int> counts = devicex->getMainGraphOpCounts();

  // for norm only, we expect all Norm ops to appear twice, all others once
  for (auto op_count : counts) {
    std::cout << "check for " << op_count.first->str() << std::endl;
    if (op_count.first->isNorm()) {
      BOOST_CHECK(op_count.second == 2);
    } else {
      BOOST_CHECK(op_count.second == 1);
    }
  }

  std::cout << devicex->getMainGraphOpString() << std::endl;
}
