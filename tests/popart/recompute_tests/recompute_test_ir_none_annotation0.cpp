#define BOOST_TEST_MODULE RecomputeTestIrNoneAnnotation0

#include <boost/test/unit_test.hpp>
#include <vector>

#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/filereader.hpp>
#include <popart/ir.hpp>
#include <popart/names.hpp>
#include <popart/op/l1.hpp>
#include <popart/optimizer.hpp>
#include <popart/sessionoptions.hpp>
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

BOOST_AUTO_TEST_CASE(NoRecomputeTest) {
  auto run_test = [](bool enableOutlining, int expectedScheduleSize) {
    // Build an onnnx model
    auto builder = Builder::create();
    auto aiOnnx  = builder->aiOnnxOpset9();

    TensorInfo input_shape{"FLOAT", std::vector<int64_t>{1, 4, 32, 32}};

    TensorInfo weights_shape{"FLOAT", std::vector<int64_t>{4, 4, 3, 3}};
    float weight_vals[4 * 4 * 3 * 3] = {0};
    ConstVoidData weight_data        = {weight_vals, weights_shape};

    auto act = builder->addInputTensor(input_shape);

    for (int i = 0; i < 7; ++i) {
      act = conv(builder.get(), act, weight_data);
      act = aiOnnx.relu({act});
    }

    auto proto      = builder->getModelProto();
    auto modelProto = io::getModelFromString(proto);

    // Add the last tensor, and the 3rd tensor as anchors
    auto dataFlow  = DataFlow(1, {{act, AnchorReturnType("ALL")}});
    auto optimizer = ConstSGD(0.01);
    std::vector<Loss *> losses{
        new L1Loss(act, "l1LossVal", 0.1, ReductionType::SUM)};
    auto cpuDevice = DeviceManager::createDeviceManager().createCpuDevice();

    SessionOptions opts;
    opts.autoRecomputation              = RecomputationType::None;
    opts.enableOutlining                = enableOutlining;
    opts.enableOutliningCopyCostPruning = false;
    opts.mergeVarUpdate                 = MergeVarUpdateType::None;

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

    // All but the original 6 operations should be pruned
    BOOST_CHECK_EQUAL(ir.getOpSchedule({}).size(), expectedScheduleSize);
  };

  run_test(false, 42);
  run_test(true, 24);
}
