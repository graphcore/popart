// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE ConstExprFloorTest

#include <boost/test/unit_test.hpp>
#include <filereader.hpp>
#include <memory>
#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/devicemanager.hpp>
#include <popart/half.hpp>
#include <popart/inputshapeinfo.hpp>
#include <popart/ir.hpp>
#include <popart/names.hpp>
#include <popart/ndarraywrapper.hpp>
#include <popart/op/identity.hpp>
#include <popart/session.hpp>
#include <popart/sgd.hpp>
#include <popart/tensordata.hpp>
#include <popart/testdevice.hpp>

using namespace popart;

namespace {

template <typename T> std::vector<char> getRawConstData() {
  //
  // matrix:
  //
  // [1.2 1.4
  //  1.6 1.8]
  //
  // of type determined by template paramater
  //
  std::vector<char> raw_const_data;
  raw_const_data.resize(sizeof(T) * 4);
  T *raw_const_ptr = reinterpret_cast<T *>(raw_const_data.data());
  for (int i = 0; i < 4; ++i) {
    raw_const_ptr[i] = static_cast<T>(1.2f + i * 0.2f);
  }
  return raw_const_data;
}

std::string getTestModelProto(DataType type) {

  // The compute graph :
  //
  // x0 --|
  //      |--- [MATMUL] -- x2 -- [FLOOR] ----------- x3 --|
  //      |                                               |
  // x1 --|                                               | -- [ADD] --x6
  //                                                      |
  // x4 const - [FLOOR] -- x5 -- [CAST to FLOAT] -- c0 ---|
  //
  // This test has 1 non-const floor and 1 const floor to perform

  ConstVoidData const_data;
  std::vector<char> raw_const_data;
  std::string type_string;
  if (type == DataType::FLOAT) {
    raw_const_data = getRawConstData<float>();
    type_string    = "FLOAT";
  }

  else if (type == DataType::INT32) {
    raw_const_data = getRawConstData<int>();
    type_string    = "INT32";
  }

  else if (type == DataType::FLOAT16) {
    raw_const_data = getRawConstData<float16_t>();
    type_string    = "FLOAT16";
  }

  else {
    throw error("this test is not enabled for speficied type");
  }

  // all 3 matrices x0, x1 & x4 are 2x2
  TensorInfo const_info = {type_string, Shape{2, 2}};
  TensorInfo in_info    = {"FLOAT", Shape{2, 2}};
  const_data            = {raw_const_data.data(), const_info};

  auto builder     = Builder::create();
  auto aiOnnx      = builder->aiOnnxOpset9();
  auto aiGraphcore = builder->aiGraphcoreOpset1();

  auto x4   = aiOnnx.constant(const_data, "x4");
  auto x0   = builder->addInputTensor(in_info);
  auto x1   = builder->addInputTensor(in_info);
  auto x2   = aiOnnx.matmul({x0, x1});
  auto x3   = aiOnnx.floor({x2});
  auto x5   = aiOnnx.floor({x4});
  auto c0   = aiOnnx.cast({x5}, "FLOAT");
  auto x6   = aiOnnx.add({x3, c0});
  auto loss = aiGraphcore.identityloss({x6});
  builder->addOutputTensor(x6);
  builder->addOutputTensor(loss);

  auto proto = builder->getModelProto();
  return proto;
}
} // namespace

// Test that the floor is correctly identified for const expression removal
BOOST_AUTO_TEST_CASE(ConstExprTest_Floor0) {

  auto proto       = getTestModelProto(DataType::FLOAT);
  auto model_proto = io::getModelFromString(proto);

  // Create the IR
  auto art       = AnchorReturnType("All");
  TensorId outId = model_proto.graph().output(1).name();
  auto data_flow = DataFlow(1, {{outId, art}});
  auto device    = createTestDevice(TEST_TARGET);

  Ir ir;
  ir.prepare({model_proto,
              InputShapeInfo(),
              data_flow,
              {},
              nullptr,
              *device,
              {}, // no SessionOptions
              Patterns({}).enableRuntimeAsserts(false)});

  // only the one floor (after the matmul) should remain
  BOOST_CHECK(ir.opsOfType(Onnx::AiOnnx::OpSet9::Floor).size() == 1);
}

// Test that the graph output is correct
BOOST_AUTO_TEST_CASE(ConstExprTest_Floor1) {

  auto numerical_accuracy_test = [](popart::DataType type) {
    popart::logging::session::info("In numerical_accuracy_test for type {}",
                                   TensorInfo(type, {1}).data_type());

    auto proto = getTestModelProto(type);

    auto model_proto = io::getModelFromString(proto);

    // Create the IR
    auto art       = AnchorReturnType("All");
    TensorId outId = model_proto.graph().output(0).name();
    auto data_flow = DataFlow(1, {{outId, art}});
    auto optimizer = ConstSGD(0.01);

    auto device = popart::createTestDevice(TEST_TARGET);

    auto session = popart::InferenceSession::createFromOnnxModel(
        proto,
        data_flow,
        device,
        popart::InputShapeInfo(),
        {}, // no session options
        Patterns(PatternsLevel::NoPatterns)
            .enableRuntimeAsserts(false) // no patterns
    );

    // prepare the anchors
    std::vector<float> rawOutputData(4);
    popart::NDArrayWrapper<float> outData(rawOutputData.data(), {2, 2});

    std::map<popart::TensorId, popart::IArray &> anchors = {
        {outId, outData},
    };

    session->prepareDevice();

    std::vector<float> rawInputData0{1.2f, 2.4f, 3.6f, 4.8f};
    std::vector<float> rawInputData1{5.2f, 6.4f, 7.6f, 8.8f};

    //
    // the calculation, done in numpy:
    //
    // In [2]: np.floor(np.dot([[1.2,2.4],[3.6,4.8]], [[5.2,6.4],[7.6,8.8]]))
    //       + np.floor(np.array([[1.2,1.4],[1.6,1.8]]))
    //
    // Out[2]:
    // array([[25., 29. ],
    //        [56., 66.. ]])
    //

    std::vector<float> baseline{25., 29., 56., 66.};

    popart::NDArrayWrapper<float> inData0(rawInputData0.data(), {2, 2});
    popart::NDArrayWrapper<float> inData1(rawInputData1.data(), {2, 2});

    std::map<popart::TensorId, popart::IArray &> inputs = {
        {model_proto.graph().input(0).name(), inData0},
        {model_proto.graph().input(1).name(), inData1}};

    popart::StepIO stepio(inputs, anchors);

    session->run(stepio);

    BOOST_CHECK(std::equal(baseline.data(),
                           baseline.data() + 4,
                           rawOutputData.data()) == true);
  };

  numerical_accuracy_test(DataType::FLOAT);
  numerical_accuracy_test(DataType::FLOAT16);
}
