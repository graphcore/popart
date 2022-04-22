// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE UnwindingTest0

#include <algorithm>
#include <boost/test/unit_test.hpp>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <popart/dataflow.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/ndarraywrapper.hpp>
#include <popart/op/matmul.hpp>
#include <popart/op/slice.hpp>
#include <popart/session.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/testdevice.hpp>

#include "popart/datatype.hpp"
#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/operators.hpp"
#include "popart/stepio.hpp"
#include "popart/tensordebuginfo.hpp"
#include "popart/tensors.hpp"

namespace popart {
class IArray;
} // namespace popart

using namespace popart;

// Test unwinding in a tricky situation, unwinding through two nested slices,
// where only one part of the tensor has a creator (MatMul) and the rest has to
// be filled linearly.
// There is currently no way to probe if the tensor was created as expected,
// so the test's success criteria is passing without throwing an exception.
BOOST_AUTO_TEST_CASE(UnwindingTest0) {
  auto ir  = std::make_unique<Ir>();
  Graph &g = ir->getMainGraph();

  const TensorInfo info_A{DataType::FLOAT, Shape{12, 1, 12, 518, 64}};
  const TensorInfo info_B{DataType::FLOAT, Shape{1, 1, 1, 64, 64}};
  const TensorInfo info_E{DataType::FLOAT, Shape{1, 1, 12, 64, 517}};

  std::vector<float> data_A(info_A.nelms(), 1);
  std::vector<float> data_B(info_B.nelms(), 1);

  g.getTensors().addVarInit("A", info_A, data_A.data());
  g.getTensors().addVarInit("D", info_B, data_B.data());

  Op::Settings gSettings(g, "op", {});

  g.createConnectedOp<SliceOp>({{SliceOp::getInIndex(), "A"}},
                               {{SliceOp::getOutIndex(), "B"}},
                               Onnx::Operators::Slice_11,
                               std::vector<int64_t>{10},
                               std::vector<int64_t>{11},
                               std::vector<int64_t>{0},
                               std::vector<int64_t>{},
                               gSettings.copy("SliceOp"));

  g.createConnectedOp<SliceOp>({{SliceOp::getInIndex(), "B"}},
                               {{SliceOp::getOutIndex(), "C"}},
                               Onnx::Operators::Slice_11,
                               std::vector<int64_t>{0},
                               std::vector<int64_t>{517},
                               std::vector<int64_t>{3},
                               std::vector<int64_t>{},
                               gSettings.copy("SliceOp"));

  g.createConnectedOp<MatMulOp>(
      {{MatMulOp::getLhsInIndex(), "C"}, {MatMulOp::getRhsInIndex(), "D"}},
      {{MatMulOp::getOutIndex(), "E"}},
      Onnx::Operators::MatMul_9,
      gSettings.copy("MatMulOp"),
      0.1,
      MatMulBaseOp::SerialiseSettings(),
      OptionalDataType());

  ir->setDataFlow(DataFlow{1, {{"E", AnchorReturnType("All")}}});
  ir->updateVertices();
  ir->setIsPrepared();

  const auto session = InferenceSession::createFromIr(
      std::move(ir), createTestDevice(TEST_TARGET));
  session->prepareDevice();

  std::vector<float> outHost(info_E.nelms());
  NDArrayWrapper<float> outWrapper(outHost.data(), info_E);
  std::map<TensorId, IArray &> anchors = {{"E", outWrapper}};

  StepIO stepio({}, anchors);
  session->weightsFromHost();
  session->run(stepio);
}
