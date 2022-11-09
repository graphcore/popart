// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE ConvTransposePatternTests
#include <boost/test/unit_test.hpp>
#include <map>

#include "popart/graph.hpp"
#include "popart/ir.hpp"
#include "popart/logging.hpp"
#include "popart/names.hpp"
#include "popart/onnxoperators.gen.hpp"
#include "popart/op/convbase.hpp"
#include "popart/op/convtranspose.hpp"
#include "popart/tensorindex.hpp"

namespace popart {
class error;
}

using namespace popart;

BOOST_AUTO_TEST_CASE(TestConvTransposeThrowsOnInvalidChannels) {
  // Deliberately set up some conv values that will throw due to
  // channel dimensions not matching on the input and weights
  int inChannel    = 10;
  int outChannel   = 20;
  int inputHeight  = 50;
  int inputWidth   = 50;
  int weightHeight = 3;
  int weightWidth  = 3;

  Shape inputShape  = {1, inChannel, inputHeight, inputWidth};
  Shape weightShape = {outChannel, inChannel, weightHeight, weightWidth};

  Ir ir;
  Graph &graph = ir.getMainGraph();

  auto convAttr = Attributes();
  auto convOpts =
      MultiConvOptions(ir.getSessionOptions().convolutionOptions, convAttr);

  Op::Settings settings(graph, "test_convtranspose");
  auto autoPad = AutoPad::NOTSET;

  Tensor dataTensor("data", popart::TensorType::ActGrad, graph);
  dataTensor.info.set(popart::DataType::FLOAT8_143, inputShape);

  Tensor weightTensor("weight", popart::TensorType::ActGrad, graph);
  weightTensor.info.set(popart::DataType::FLOAT8_152, weightShape);

  Tensor out("output", popart::TensorType::ActGrad, graph);

  // Create a conv transpose op
  popart::ConvTransposeOp convTransposeOp(Onnx::Operators::Conv_1,
                                          settings,
                                          std::vector<int64_t>(),
                                          std::vector<int64_t>(),
                                          std::vector<int64_t>(),
                                          1,
                                          autoPad,
                                          std::vector<int64_t>(),
                                          std::vector<int64_t>(),
                                          convOpts);

  convTransposeOp.input->insert(0, &dataTensor);
  convTransposeOp.input->insert(1, &weightTensor);
  convTransposeOp.output->insert(0, &out);

  auto checkError = [&inputShape](popart::error const &error) {
    auto expected = logging::format(
        "Unexpected number of channels in the input tensor: {}", inputShape[1]);
    return std::string(error.what()).find(expected) != std::string::npos;
  };
  BOOST_CHECK_EXCEPTION(convTransposeOp.setup(), error, checkError);
}
