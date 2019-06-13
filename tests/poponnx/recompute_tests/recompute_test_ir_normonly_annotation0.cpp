#define BOOST_TEST_MODULE RecomputeTestIrNormOnlyAnnotation0

#include <boost/test/unit_test.hpp>
#include <vector>

#include <poponnx/builder.hpp>
#include <poponnx/dataflow.hpp>
#include <poponnx/filereader.hpp>
#include <poponnx/ir.hpp>
#include <poponnx/names.hpp>
#include <poponnx/op/l1.hpp>
#include <poponnx/optimizer.hpp>
#include <poponnx/optionflags.hpp>
#include <poponnx/tensordata.hpp>

using namespace poponnx;

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

BOOST_AUTO_TEST_CASE(NormOnlyRecomputeTest) {

  // Test that norms are RECOMPUTE

  // The model:
  //
  // In -> Conv -> BN -> Relu -> Conv -> Relu -> Conv -> BN -> Out
  //

  // Build an onnx model
  auto builder = Builder::create();
  auto aiOnnx  = builder->aiOnnxOpset9();

  TensorInfo input_shape{"FLOAT", std::vector<int64_t>{1, 4, 32, 32}};

  TensorInfo weights_shape{"FLOAT", std::vector<int64_t>{4, 4, 3, 3}};
  float weight_vals[4 * 4 * 3 * 3] = {0};
  ConstVoidData weight_data        = {weight_vals, weights_shape};

  TensorInfo bn_shape{"FLOAT", std::vector<int64_t>{4}};
  float bn_vals[4]      = {0};
  ConstVoidData bn_data = {bn_vals, bn_shape};

  auto act = builder->addInputTensor(input_shape);

  act = conv(builder.get(), act, weight_data);
  act = batchnormalization(builder.get(), act, bn_data); // BN is recomputed
  act = aiOnnx.relu({act});
  act = conv(builder.get(), act, weight_data);
  act = aiOnnx.relu({act});
  act = conv(builder.get(), act, weight_data);
  act = batchnormalization(builder.get(), act, bn_data); // BN is recomputed

  auto proto      = builder->getModelProto();
  auto modelProto = io::getModelFromString(proto);

  // Add the last tensor, and the 3rd tensor as anchors
  auto dataFlow  = DataFlow(1, {{act, AnchorReturnType("ALL")}});
  auto optimizer = ConstSGD(0.01);
  std::vector<Loss *> losses{
      new L1Loss(act, "l1LossVal", 0.1, ReductionType::SUM)};
  auto cpuDevice = DeviceManager::createDeviceManager().createCpuDevice();

  SessionOptions opts;
  opts.autoRecomputation = RecomputationType::NormOnly;
  opts.enableOutlining   = false;
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

  for (auto op : ir.getOpSchedule({})) {
    if (op->isNorm()) {
      BOOST_CHECK(op->settings.recomputeType == RecomputeType::RECOMPUTE);
    } else {
      BOOST_CHECK(op->settings.recomputeType == RecomputeType::CHECKPOINT);
    }
  }
}
