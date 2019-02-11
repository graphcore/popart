#define BOOST_TEST_MODULE ConstExprScaleTest

#include <boost/test/unit_test.hpp>
#include <poponnx/builder.hpp>
#include <poponnx/dataflow.hpp>
#include <poponnx/devicemanager.hpp>
#include <poponnx/filereader.hpp>
#include <poponnx/inputshapeinfo.hpp>
#include <poponnx/ir.hpp>
#include <poponnx/names.hpp>
#include <poponnx/ndarraywrapper.hpp>
#include <poponnx/op/l1.hpp>
#include <poponnx/optimizer.hpp>
#include <poponnx/session.hpp>
#include <poponnx/tensordata.hpp>

using namespace poponnx;

namespace {

template <typename T> std::vector<char> getRawConstData() {
  //
  // matrix:
  //
  // [2 4
  //  6 8]
  //
  // of type determined by template paramater
  //
  std::vector<char> raw_const_data;
  raw_const_data.resize(sizeof(T) * 4);
  T *raw_const_ptr = reinterpret_cast<T *>(raw_const_data.data());
  for (int i = 0; i < 4; ++i) {
    raw_const_ptr[i] = 2 * static_cast<T>(i) + 2.0f;
  }
  return raw_const_data;
}

std::string getTestModelProto(DataType type) {

  // The compute graph :
  //
  // x0 --|
  //      |--- [MATMUL] -- x2 -- [SCALE] ----------- x3 --|
  //      |                                               |
  // x1 --|                                               | -- [ADD] --x6
  //                                                      |
  // x4 const - [SCALE] -- x5 -- [CAST to FLOAT] -- c0 ---|
  //
  // This test has 1 non-const scale and 1 const scale to perform

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

  else {
    throw error("this test is not enabled for speficied type");
  }

  // all 3 matrices x0, x1 & x4 are 2x2
  TensorInfo const_info = {type_string, Shape{2, 2}};
  TensorInfo in_info    = {"FLOAT", Shape{2, 2}};
  const_data            = {raw_const_data.data(), const_info};

  auto builder = Builder::create();
  auto aiOnnx  = builder->aiOnnxOpset9();

  auto x4 = aiOnnx.constant(const_data, "x4");
  auto x0 = builder->addInputTensor(in_info);
  auto x1 = builder->addInputTensor(in_info);
  auto x2 = aiOnnx.matmul({x0, x1});
  auto x3 = aiOnnx.scale({x2}, 0.5);
  auto x5 = aiOnnx.scale({x4}, 0.5);
  auto c0 = aiOnnx.cast({x5}, DataType::FLOAT);
  auto x6 = aiOnnx.add({x3, c0});
  builder->addOutputTensor(x6);

  auto proto = builder->getModelProto();
  return proto;
}
} // namespace

// Test that the scale is correctly identified for const expression removal
BOOST_AUTO_TEST_CASE(ConstExprTest_Scale0) {

  auto proto       = getTestModelProto(DataType::FLOAT);
  auto model_proto = io::getModelFromString(proto);

  // Create the IR
  auto art       = AnchorReturnType("ALL");
  TensorId outId = model_proto.graph().output(0).name();
  auto data_flow = DataFlow(1, {{outId, art}});
  auto optimizer = ConstSGD(0.01);
  std::vector<Loss *> losses{new L1Loss(outId, "l1LossVal", 0.1)};

  Ir ir;
  ir.prepare({model_proto,
              InputShapeInfo(),
              data_flow,
              losses,
              &optimizer,
              {}, // no SessionOptions
              Patterns({})});

  // only the one scale (after the matmul) should remain
  BOOST_CHECK(ir.opsOfType(Onnx::AiOnnx::OpSet9::Scale).size() == 1);
}

// Test that the graph output is correct
BOOST_AUTO_TEST_CASE(ConstExprTest_Scale1) {

  // we will run this test twice, once when the constant to be
  // scaled is INT32, and once when it is FLOAT
  auto numerical_accuracy_test = [](poponnx::DataType type) {
    poponnx::logging::session::info("In numerical_accuracy_test for type {}",
                                    TensorInfo(type, {1}).data_type());

    auto proto = getTestModelProto(type);

    auto model_proto = io::getModelFromString(proto);

    // Create the IR
    auto art       = AnchorReturnType("ALL");
    TensorId outId = model_proto.graph().output(0).name();
    auto data_flow = DataFlow(1, {{outId, art}});
    auto optimizer = ConstSGD(0.01);
    std::vector<Loss *> losses{};

    auto session =
        poponnx::Session::createFromOnnxModel(proto,
                                              data_flow,
                                              poponnx::InputShapeInfo(),
                                              {},        // no losses
                                              nullptr,   // no optimizer
                                              {},        // no session options
                                              Patterns() // no patterns
        );

    auto cpuDevice =
        poponnx::DeviceManager::createDeviceManager().createCpuDevice();
    session->setDevice(*cpuDevice);

    // prepare the anchors
    std::vector<float> rawOutputData(4);
    poponnx::NDArrayWrapper<float> outData(rawOutputData.data(), {2, 2});

    std::map<poponnx::TensorId, poponnx::IArray &> anchors = {
        {outId, outData},
    };

    session->prepareDevice();

    std::vector<float> rawInputData0{1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> rawInputData1{5.0f, 6.0f, 7.0f, 8.0f};

    //
    // the calculation, done in numpy:
    //
    // In [2]: 0.5*np.dot([[1,2],[3,4]], [[5,6],[7,8]])
    //       + 0.5*np.array([[2,4],[6,8]])
    //
    // Out[2]:
    // array([[10.5, 13. ],
    //        [24.5, 29. ]])
    //

    std::vector<float> baseline{10.5, 13, 24.5, 29};

    poponnx::NDArrayWrapper<float> inData0(rawInputData0.data(), {2, 2});
    poponnx::NDArrayWrapper<float> inData1(rawInputData1.data(), {2, 2});

    std::map<poponnx::TensorId, poponnx::IArray &> inputs = {
        {model_proto.graph().input(0).name(), inData0},
        {model_proto.graph().input(1).name(), inData1}};

    poponnx::StepIO stepio(inputs, anchors);

    session->infer(stepio);

    BOOST_CHECK(std::equal(baseline.data(),
                           baseline.data() + 4,
                           rawOutputData.data()) == true);
  };

  numerical_accuracy_test(DataType::INT32);
  numerical_accuracy_test(DataType::FLOAT);
}
