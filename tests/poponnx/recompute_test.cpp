#define BOOST_TEST_MODULE PatternsTest

#include <boost/test/unit_test.hpp>
#include <vector>

#include <poponnx/builder.hpp>
#include <poponnx/dataflow.hpp>
#include <poponnx/filereader.hpp>
#include <poponnx/inputshapeinfo.hpp>
#include <poponnx/ir.hpp>
#include <poponnx/op/l1.hpp>
#include <poponnx/optimizer.hpp>
#include <poponnx/tensor.hpp>
#include <poponnx/tensordata.hpp>
#include <poponnx/tensorinfo.hpp>
#include <poponnx/tensors.hpp>

using namespace poponnx;

BOOST_AUTO_TEST_CASE(NoRecomputeTest) {
  // Build an onnnx model
  auto builder = Builder::create();
  auto aiOnnx  = builder->aiOnnxOpset9();

  TensorInfo input_shape{"FLOAT", std::vector<int64_t>{1, 4, 32, 32}};

  TensorInfo weights_shape{"FLOAT", std::vector<int64_t>{4, 4, 3, 3}};
  float weight_vals[4 * 4 * 3 * 3] = {0};
  ConstVoidData weight_data        = {weight_vals, weights_shape};

  auto act = builder->addInputTensor(input_shape);

  auto weights = builder->addInitializedInputTensor(weight_data);
  act = aiOnnx.conv({act, weights}, {1, 1}, 1, {}, {1, 1, 1, 1}, {1, 1});
  act = aiOnnx.relu({act});

  weights = builder->addInitializedInputTensor(weight_data);
  act     = aiOnnx.conv({act, weights}, {1, 1}, 1, {}, {1, 1, 1, 1}, {1, 1});
  act     = aiOnnx.relu({act});

  weights = builder->addInitializedInputTensor(weight_data);
  act     = aiOnnx.conv({act, weights}, {1, 1}, 1, {}, {1, 1, 1, 1}, {1, 1});
  act     = aiOnnx.relu({act});

  weights = builder->addInitializedInputTensor(weight_data);
  act     = aiOnnx.conv({act, weights}, {1, 1}, 1, {}, {1, 1, 1, 1}, {1, 1});
  act     = aiOnnx.relu({act});

  weights = builder->addInitializedInputTensor(weight_data);
  act     = aiOnnx.conv({act, weights}, {1, 1}, 1, {}, {1, 1, 1, 1}, {1, 1});
  act     = aiOnnx.relu({act});

  weights = builder->addInitializedInputTensor(weight_data);
  act     = aiOnnx.conv({act, weights}, {1, 1}, 1, {}, {1, 1, 1, 1}, {1, 1});
  act     = aiOnnx.relu({act});

  weights = builder->addInitializedInputTensor(weight_data);
  act     = aiOnnx.conv({act, weights}, {1, 1}, 1, {}, {1, 1, 1, 1}, {1, 1});
  act     = aiOnnx.relu({act});

  auto proto      = builder->getModelProto();
  auto modelProto = io::getModelFromString(proto);

  // Add the last tensor, and the 3rd tensor as anchors
  auto dataFlow  = DataFlow(1, {{act, AnchorReturnType("ALL")}});
  auto optimizer = ConstSGD(0.01);
  std::vector<Loss *> losses{new L1Loss(act, "l1LossVal", 0.1)};

  SessionOptions opts;
  opts.enableAutoRecomputation = false;

  Ir ir;
  ir.prepare({modelProto,
              InputShapeInfo(),
              dataFlow,
              losses,
              &optimizer,
              opts,
              Patterns({PreAliasPatternType::OPTOIDENTITY,
                        PreAliasPatternType::POSTNREPL})});

  // All but the original 6 operations should be pruned
  BOOST_CHECK_EQUAL(ir.getOpSchedule({}).size(), 42);
}

BOOST_AUTO_TEST_CASE(RecomputeTest) {
  // Build an onnnx model
  auto builder = Builder::create();
  auto aiOnnx  = builder->aiOnnxOpset9();

  TensorInfo input_shape{"FLOAT", std::vector<int64_t>{1, 4, 32, 32}};

  TensorInfo weights_shape{"FLOAT", std::vector<int64_t>{4, 4, 3, 3}};
  float weight_vals[4 * 4 * 3 * 3] = {0};
  ConstVoidData weight_data        = {weight_vals, weights_shape};

  auto act = builder->addInputTensor(input_shape);

  auto weights = builder->addInitializedInputTensor(weight_data);
  act = aiOnnx.conv({act, weights}, {1, 1}, 1, {}, {1, 1, 1, 1}, {1, 1});
  act = aiOnnx.relu({act});

  weights = builder->addInitializedInputTensor(weight_data);
  act     = aiOnnx.conv({act, weights}, {1, 1}, 1, {}, {1, 1, 1, 1}, {1, 1});
  act     = aiOnnx.relu({act});

  weights = builder->addInitializedInputTensor(weight_data);
  act     = aiOnnx.conv({act, weights}, {1, 1}, 1, {}, {1, 1, 1, 1}, {1, 1});
  act     = aiOnnx.relu({act});

  weights = builder->addInitializedInputTensor(weight_data);
  act     = aiOnnx.conv({act, weights}, {1, 1}, 1, {}, {1, 1, 1, 1}, {1, 1});
  act     = aiOnnx.relu({act});

  weights = builder->addInitializedInputTensor(weight_data);
  act     = aiOnnx.conv({act, weights}, {1, 1}, 1, {}, {1, 1, 1, 1}, {1, 1});
  act     = aiOnnx.relu({act});

  weights = builder->addInitializedInputTensor(weight_data);
  act     = aiOnnx.conv({act, weights}, {1, 1}, 1, {}, {1, 1, 1, 1}, {1, 1});
  act     = aiOnnx.relu({act});

  weights = builder->addInitializedInputTensor(weight_data);
  act     = aiOnnx.conv({act, weights}, {1, 1}, 1, {}, {1, 1, 1, 1}, {1, 1});
  act     = aiOnnx.relu({act});

  auto proto      = builder->getModelProto();
  auto modelProto = io::getModelFromString(proto);

  // Add the last tensor, and the 3rd tensor as anchors
  auto dataFlow  = DataFlow(1, {{act, AnchorReturnType("ALL")}});
  auto optimizer = ConstSGD(0.01);
  std::vector<Loss *> losses{new L1Loss(act, "l1LossVal", 0.1)};

  SessionOptions opts;
  opts.enableAutoRecomputation = true;

  Ir ir;
  ir.prepare({modelProto,
              InputShapeInfo(),
              dataFlow,
              losses,
              &optimizer,
              opts,
              Patterns({PreAliasPatternType::OPTOIDENTITY,
                        PreAliasPatternType::POSTNREPL})});

  // All but the original 6 operations should be pruned
  BOOST_CHECK_EQUAL(ir.getOpSchedule({}).size(), 46);
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

  SessionOptions opts;
  opts.enableRecomputation = true;
  opts.dotChecks           = {DotCheck::FINAL};

  Ir ir;
  ir.prepare({modelProto,
              InputShapeInfo(),
              dataFlow,
              losses,
              &optimizer,
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
