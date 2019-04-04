#define BOOST_TEST_MODULE PatternsTest

#include <boost/test/unit_test.hpp>
#include <vector>

#include <poponnx/attributes.hpp>
#include <poponnx/builder.hpp>
#include <poponnx/dataflow.hpp>
#include <poponnx/filereader.hpp>
#include <poponnx/inputshapeinfo.hpp>
#include <poponnx/ir.hpp>
#include <poponnx/op.hpp>
#include <poponnx/op/batchnorm.hpp>
#include <poponnx/op/div.hpp>
#include <poponnx/op/groupnorm.hpp>
#include <poponnx/op/instancenorm.hpp>
#include <poponnx/op/l1.hpp>
#include <poponnx/op/relu.hpp>
#include <poponnx/op/sigmoid.hpp>
#include <poponnx/op/softmax.hpp>
#include <poponnx/op/tanh.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/optimizer.hpp>
#include <poponnx/tensor.hpp>
#include <poponnx/tensordata.hpp>
#include <poponnx/tensorinfo.hpp>
#include <poponnx/tensors.hpp>
#include <poponnx/transforms/recompute.hpp>

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

BOOST_AUTO_TEST_CASE(NoRecomputeTest) {
  // Build an onnnx model
  auto builder = Builder::create();
  auto aiOnnx  = builder->aiOnnxOpset9();

  TensorInfo input_shape{"FLOAT", std::vector<int64_t>{1, 4, 32, 32}};

  TensorInfo weights_shape{"FLOAT", std::vector<int64_t>{4, 4, 3, 3}};
  float weight_vals[4 * 4 * 3 * 3] = {0};
  ConstVoidData weight_data        = {weight_vals, weights_shape};

  auto act = builder->addInputTensor(input_shape);

  act = conv(builder.get(), act, weight_data);
  act = aiOnnx.relu({act});

  act = conv(builder.get(), act, weight_data);
  act = aiOnnx.relu({act});

  act = conv(builder.get(), act, weight_data);
  act = aiOnnx.relu({act});

  act = conv(builder.get(), act, weight_data);
  act = aiOnnx.relu({act});

  act = conv(builder.get(), act, weight_data);
  act = aiOnnx.relu({act});

  act = conv(builder.get(), act, weight_data);
  act = aiOnnx.relu({act});

  act = conv(builder.get(), act, weight_data);
  act = aiOnnx.relu({act});

  auto proto      = builder->getModelProto();
  auto modelProto = io::getModelFromString(proto);

  // Add the last tensor, and the 3rd tensor as anchors
  auto dataFlow  = DataFlow(1, {{act, AnchorReturnType("ALL")}});
  auto optimizer = ConstSGD(0.01);
  std::vector<Loss *> losses{new L1Loss(act, "l1LossVal", 0.1)};
  auto cpuDevice = DeviceManager::createDeviceManager().createCpuDevice();

  SessionOptions opts;
  opts.autoRecomputation = RecomputationType::None;

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
  BOOST_CHECK_EQUAL(ir.getOpSchedule({}).size(), 42);
}

BOOST_AUTO_TEST_CASE(StandardRecomputeTest) {
  // Build an onnnx model
  auto builder = Builder::create();
  auto aiOnnx  = builder->aiOnnxOpset9();

  TensorInfo input_shape{"FLOAT", std::vector<int64_t>{1, 4, 32, 32}};

  TensorInfo weights_shape{"FLOAT", std::vector<int64_t>{4, 4, 3, 3}};
  float weight_vals[4 * 4 * 3 * 3] = {0};
  ConstVoidData weight_data        = {weight_vals, weights_shape};

  auto act = builder->addInputTensor(input_shape);

  act = conv(builder.get(), act, weight_data);
  act = aiOnnx.relu({act});

  act = conv(builder.get(), act, weight_data);
  act = aiOnnx.relu({act});

  act = conv(builder.get(), act, weight_data);
  act = aiOnnx.relu({act});

  act = conv(builder.get(), act, weight_data);
  act = aiOnnx.relu({act});

  act = conv(builder.get(), act, weight_data);
  act = aiOnnx.relu({act});

  act = conv(builder.get(), act, weight_data);
  act = aiOnnx.relu({act});

  act = conv(builder.get(), act, weight_data);
  act = aiOnnx.relu({act});

  auto proto      = builder->getModelProto();
  auto modelProto = io::getModelFromString(proto);

  // Add the last tensor, and the 3rd tensor as anchors
  auto dataFlow  = DataFlow(1, {{act, AnchorReturnType("ALL")}});
  auto optimizer = ConstSGD(0.01);
  std::vector<Loss *> losses{new L1Loss(act, "l1LossVal", 0.1)};
  auto cpuDevice = DeviceManager::createDeviceManager().createCpuDevice();

  SessionOptions opts;
  opts.autoRecomputation = RecomputationType::Standard;

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
  BOOST_CHECK_EQUAL(ir.getOpSchedule({}).size(), 46);
}

BOOST_AUTO_TEST_CASE(NormOnlyRecomputeTest) {
  // Test that norms (and non-linearities following norms) are cloned in the
  // ir for recomputation in the backwards pass.

  // The model:
  //
  // In -> Conv -> BN -> Relu -> Conv -> Relu -> Conv -> BN -> Out
  //
  // With RecomputationType::None:
  //   BN: 2
  //   Relu: 2
  // With RecomputationType::NormOnly:
  //   BN: 4 (both recomputed)
  //   Relu: 3 (recomputed only when following norm)

  // Build an onnnx model
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
  act = aiOnnx.relu({act}); // Relu after Batchnorm is recomputed
  act = conv(builder.get(), act, weight_data);
  act = aiOnnx.relu({act}); // Relu after conv is not recomputed
  act = conv(builder.get(), act, weight_data);
  act = batchnormalization(builder.get(), act, bn_data); // BN is recomputed

  auto proto      = builder->getModelProto();
  auto modelProto = io::getModelFromString(proto);

  // Add the last tensor, and the 3rd tensor as anchors
  auto dataFlow  = DataFlow(1, {{act, AnchorReturnType("ALL")}});
  auto optimizer = ConstSGD(0.01);
  std::vector<Loss *> losses{new L1Loss(act, "l1LossVal", 0.1)};
  auto cpuDevice = DeviceManager::createDeviceManager().createCpuDevice();

  SessionOptions opts;
  opts.autoRecomputation = RecomputationType::NormOnly;

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

  BOOST_CHECK(ir.opsOfType(Onnx::AiOnnx::OpSet9::BatchNormalization).size() ==
              4);
  BOOST_CHECK(ir.opsOfType(Onnx::AiOnnx::OpSet9::Relu).size() == 3);
}

BOOST_AUTO_TEST_CASE(DontInheritRecomputeTest) {
  // Build an onnnx model
  auto builder = Builder::create();
  auto aiOnnx  = builder->aiOnnxOpset9();

  TensorInfo input_shape{"FLOAT", std::vector<int64_t>{1, 4, 32, 32}};

  auto act = builder->addInputTensor(input_shape);

  auto relu_out = aiOnnx.relu({act});

  builder->recomputeOutputInBackwardPass(relu_out);

  auto proto      = builder->getModelProto();
  auto modelProto = io::getModelFromString(proto);

  // Add the last tensor, and the 3rd tensor as anchors
  auto dataFlow  = DataFlow(1,
                           {{relu_out, AnchorReturnType("ALL")},
                            {"d__" + act, AnchorReturnType("ALL")}});
  auto optimizer = ConstSGD(0.01);
  std::vector<Loss *> losses{new L1Loss(relu_out, "l1LossVal", 0.1)};
  auto cpuDevice = DeviceManager::createDeviceManager().createCpuDevice();

  SessionOptions opts;
  opts.dotChecks = {DotCheck::FINAL};

  Ir ir;
  ir.prepare({modelProto,
              InputShapeInfo(),
              dataFlow,
              losses,
              &optimizer,
              *cpuDevice,
              opts,
              Patterns()});

  // Check the relu op has recomputation enabled
  auto tensor = ir.getTensors().get(relu_out);
  auto op     = tensor->getProducer();
  // check we have the correct op
  BOOST_CHECK(op->opid.type == "Relu");
  BOOST_CHECK(op->getRecomputeOutput() != boost::none &&
              *op->getRecomputeOutput() == true);

  // Check the grad op has not inherited the recomputation
  auto grad_tensor = ir.getTensors().get("d__" + act);
  auto grad_op     = grad_tensor->getProducer();
  // check we have the correct op
  BOOST_CHECK(grad_op->opid.type == "ReluGrad");
  BOOST_CHECK(grad_op->getRecomputeOutput() == boost::none);
}

BOOST_AUTO_TEST_CASE(IsNormTest) {
  poponnx::Ir ir;

  // Is 'add' a norm? - No
  std::unique_ptr<Op> add = OpManager::createOp(Onnx::Operators::Add_7, ir);
  BOOST_CHECK(!add.get()->isNorm());

  // Is 'batchnorm' a norm? - Yes
  std::unique_ptr<Op> bn =
      OpManager::createOp(Onnx::Operators::BatchNormalization_9, ir);
  BOOST_CHECK(bn.get()->isNorm());

  // Is 'groupnorm' a norm? - Yes
  Node node;
  node.add_attribute();
  node.mutable_attribute(0)->set_name("num_groups");
  node.mutable_attribute(0)->set_i(1);
  NodeAttributes nodeAttr = node.attribute();
  Attributes attr(nodeAttr);
  std::unique_ptr<Op> gn = OpManager::createOp(
      Onnx::CustomOperators::GroupNormalization_1, ir, "", attr);
  BOOST_CHECK(gn.get()->isNorm());

  // Is 'instancenorm' a norm? - Yes
  std::unique_ptr<Op> in =
      OpManager::createOp(Onnx::Operators::InstanceNormalization_6, ir);
  BOOST_CHECK(in.get()->isNorm());
}

BOOST_AUTO_TEST_CASE(IsNonLinearityTest) {
  poponnx::Ir ir;

  // Is 'div' a non-linearity? - No
  std::unique_ptr<Op> div = OpManager::createOp(Onnx::Operators::Div_7, ir);
  BOOST_CHECK(!div.get()->isNonlinearity());

  // Is 'tanh' a non-linearity? - Yes
  std::unique_ptr<Op> tanh = OpManager::createOp(Onnx::Operators::Tanh_6, ir);
  BOOST_CHECK(tanh.get()->isNonlinearity());

  // Is 'softmax' a non-linearity? - Yes
  std::unique_ptr<Op> sfm = OpManager::createOp(Onnx::Operators::Softmax_1, ir);
  BOOST_CHECK(sfm.get()->isNonlinearity());

  // Is 'relu' a non-linearity? - Yes
  std::unique_ptr<Op> relu = OpManager::createOp(Onnx::Operators::Relu_6, ir);
  BOOST_CHECK(relu.get()->isNonlinearity());

  // Is 'sigmoid' a non-linearity? - Yes
  std::unique_ptr<Op> sgm = OpManager::createOp(Onnx::Operators::Sigmoid_6, ir);
  BOOST_CHECK(sgm.get()->isNonlinearity());
}
