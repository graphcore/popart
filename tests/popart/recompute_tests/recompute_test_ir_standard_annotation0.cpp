#define BOOST_TEST_MODULE RecomputeTestIrStandardAnnotation0

#include <boost/test/unit_test.hpp>
#include <vector>

#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/filereader.hpp>
#include <popart/ir.hpp>
#include <popart/names.hpp>
#include <popart/op/l1.hpp>
#include <popart/optimizer.hpp>
#include <popart/optionflags.hpp>
#include <popart/tensordata.hpp>

using namespace popart;

TensorId conv(Builder *b, TensorId act, ConstVoidData wdata) {
  auto aiOnnx  = b->aiOnnxOpset9();
  auto weights = b->addInitializedInputTensor(wdata);
  act = aiOnnx.conv({act, weights}, {1, 1}, 1, {}, {1, 1, 1, 1}, {1, 1});
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

BOOST_AUTO_TEST_CASE(StandardRecomputeTest) {
  auto run_test = [](bool enableOutlining) {
    // Build an onnnx model
    auto builder = Builder::create();
    auto aiOnnx  = builder->aiOnnxOpset9();

    TensorInfo input_shape{"FLOAT", std::vector<int64_t>{1, 4, 32, 32}};

    TensorInfo weights_shape{"FLOAT", std::vector<int64_t>{4, 4, 3, 3}};
    float weight_vals[4 * 4 * 3 * 3] = {0};
    ConstVoidData weight_data        = {weight_vals, weights_shape};

    auto act = builder->addInputTensor(input_shape);

    for (int i = 0; i < 16; ++i) {
      act = conv(builder.get(), act, weight_data);
      act = aiOnnx.relu({act});
    }

    auto proto      = builder->getModelProto();
    auto modelProto = io::getModelFromString(proto);

    // Add the last tensor as an anchor
    auto dataFlow  = DataFlow(1, {{act, AnchorReturnType("ALL")}});
    auto optimizer = ConstSGD(0.01);
    std::vector<Loss *> losses{
        new L1Loss(act, "l1LossVal", 0.1, ReductionType::SUM)};
    auto cpuDevice = DeviceManager::createDeviceManager().createCpuDevice();

    SessionOptions opts;
    opts.autoRecomputation = RecomputationType::Standard;
    opts.enableOutlining   = enableOutlining;
    opts.mergeVarUpdate    = MergeVarUpdateType::None;

    Ir ir;
    ir.prepare({modelProto,
                InputShapeInfo(),
                dataFlow,
                losses,
                &optimizer,
                *cpuDevice,
                opts,
                Patterns({PreAliasPatternType::OPTOIDENTITY,
                          PreAliasPatternType::POSTNREPL})});

    int nRecompute = 0;
    for (auto op : ir.getOpSchedule({})) {
      if (op->settings.recomputeType == RecomputeType::RECOMPUTE) {
        ++nRecompute;
      }
    }

    std::cout << "with enableOutlining = " << enableOutlining
              << ", n recompute = " << nRecompute << std::endl;
    BOOST_CHECK(nRecompute > 0);
  };

  run_test(false);
  run_test(true);
}
