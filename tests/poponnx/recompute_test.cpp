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
#include <poponnx/tensordata.hpp>
#include <poponnx/tensorinfo.hpp>

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
  opts.enableRecomputation = false;

  Ir ir;
  ir.prepare({modelProto,
              InputShapeInfo(),
              dataFlow,
              losses,
              &optimizer,
              opts,
              Patterns({PatternType::OPTOIDENTITY, PatternType::POSTNREPL})});

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
  opts.enableRecomputation = true;

  Ir ir;
  ir.prepare({modelProto,
              InputShapeInfo(),
              dataFlow,
              losses,
              &optimizer,
              opts,
              Patterns({PatternType::OPTOIDENTITY, PatternType::POSTNREPL})});

  // All but the original 6 operations should be pruned
  BOOST_CHECK_EQUAL(ir.getOpSchedule({}).size(), 46);
}
