// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE Test_StepIO_DeonnxingRegressionTests
#include <boost/algorithm/string/predicate.hpp>
#include <boost/test/unit_test.hpp>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <popart/dataflow.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/names.hpp>
#include <popart/ndarraywrapper.hpp>
#include <popart/popx/irlowering.hpp>
#include <popart/stepio.hpp>
#include <popart/tensorinfo.hpp>

#include "popart/datatype.hpp"
#include "popart/devicemanager.hpp"
#include "popart/error.hpp"
#include "popart/popx/executablex.hpp"
#include "popart/tensordebuginfo.hpp"
#include "popart/tensors.hpp"

namespace popart {
class IArray;
} // namespace popart

using namespace popart;

namespace {

template <typename Ex>
std::function<bool(const Ex &)>
checkErrorMsgHasPrefixFn(const std::string &prefix) {
  return [=](const Ex &ex) -> bool {
    return boost::algorithm::starts_with(ex.what(), prefix);
  };
}

} // namespace

BOOST_AUTO_TEST_CASE(TestAssertNumElementsNoOnnxModel) {
  Ir ir;
  Graph &g      = ir.getMainGraph();
  const auto di = DeviceManager::createDeviceManager().createCpuDevice();

  const TensorInfo xInfo{DataType::FLOAT, Shape{2, 2}};
  const int bps = 2;

  // Build a simple graph with one stream tensor.
  g.getTensors().addStream("x", xInfo);
  ir.setDataFlow(DataFlow(bps));

  // Lower the IR.
  auto lowering = std::make_unique<popx::IrLowering>(ir, di);
  auto exe      = popx::Executablex::createFromLoweredIr(*lowering);

  // Verify that the IR has no ONNX model.
  BOOST_REQUIRE(!exe->ir().hasOnnxModel());

  // All graph inputs given and correct size. Shouldn't throw.
  {
    const TensorInfo inInfo{DataType::FLOAT, Shape{bps, 2, 2}};
    std::vector<float> xHost(inInfo.nelms(), 0);
    NDArrayWrapper<float> xIn(xHost.data(), inInfo);
    std::map<TensorId, IArray &> inputs = {{"x", xIn}};
    StepIO stepio(inputs, {});
    BOOST_REQUIRE_NO_THROW(stepio.assertNumElements(*exe));
  }

  // Input has wrong batches per step dimension (i.e. wrong size). Should throw.
  {
    const TensorInfo inInfo{DataType::FLOAT, Shape{1, 2, 2}};
    std::vector<float> xHost(inInfo.nelms(), 0);
    NDArrayWrapper<float> xIn(xHost.data(), inInfo);
    std::map<TensorId, IArray &> inputs = {{"x", xIn}};
    StepIO stepio(inputs, {});
    const auto checkErrorFn = checkErrorMsgHasPrefixFn<error>(
        "Unexpected number of input elements for Tensor x. Expected 8, but "
        "received 4.");
    BOOST_REQUIRE_EXCEPTION(
        stepio.assertNumElements(*exe), error, checkErrorFn);
  }

  // Input not in the graph. Should throw.
  {
    const TensorInfo inInfo{DataType::FLOAT, Shape{1, 2, 2}};
    std::vector<float> xHost(inInfo.nelms(), 0);
    NDArrayWrapper<float> xIn(xHost.data(), inInfo);
    std::map<TensorId, IArray &> inputs = {{"y", xIn}};
    StepIO stepio(inputs, {});
    const auto checkErrorFn = checkErrorMsgHasPrefixFn<error>(
        "Testing that the buffer provided by user for input Tensor y has the "
        "correct number of elements, But there is no Tensor named y in the "
        "Ir's main Graph.");
    BOOST_REQUIRE_EXCEPTION(
        stepio.assertNumElements(*exe), error, checkErrorFn);
  }
}
