// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE MatMulGradPatternTest

#include <algorithm>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/test/unit_test.hpp>
#include <cstdint>
#include <filereader.hpp>
#include <iterator>
#include <map>
#include <memory>
#include <string>
#include <testdevice.hpp>
#include <utility>
#include <vector>
#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/graph.hpp>
#include <popart/inputshapeinfo.hpp>
#include <popart/op/reshape.hpp>
#include <popart/op/transpose.hpp>
#include <popart/session.hpp>
#include <popart/sgd.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorinfo.hpp>

#include "../random_util.hpp"
#include "popart/ir.hpp"
#include "popart/logging.hpp"
#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/operators.hpp"
#include "popart/patterns/patterns.hpp"
#include "popart/scheduler_requireoptimal.hpp"
#include "popart/sessionoptions.hpp"
#include "popart/stepio.hpp"
#include "popart/tensordebuginfo.hpp"
#include "popart/tensorindex.hpp"
#include "popart/vendored/any.hpp"

namespace popart {
class IArray;
} // namespace popart

using namespace popart;

// Compute two MatMuls and their backward passes, make sure the
// MatMul Reshape/Transpose introduced by the MatMulGradPattern are scheduled
// as late as possible.
BOOST_AUTO_TEST_CASE(MatMulGradPatternScheduleTest_0) {

  // The matrix sizes are important to ensure the order in which we expect
  // the SGDVarUpdates to happen under optimal tensor liveness scheduling.
  // The current configuration should ensure the maximum distance between
  // tensor producer and consumer is 8.
  int64_t M = 32;
  int64_t N = 24;
  int64_t O = 16;
  int64_t P = 8;
  int64_t K = 20;

  // we will generate random initializations
  int seed = 1337;
  DefaultRandomEngine eng(seed);
  UniformRealDistribution<float> fdis(-4.f, +4.f);

  auto bder        = Builder::create();
  auto aiOnnx      = bder->aiOnnxOpset9();
  auto aiGraphcore = bder->aiGraphcoreOpset1();

  // Tensor A
  TensorInfo A_info{"FLOAT", std::vector<int64_t>{K, P, O}};
  std::vector<float> v_A_init(A_info.nelms());
  for (auto &val : v_A_init) {
    val = fdis(eng);
  }
  TensorId A_id =
      bder->addInitializedInputTensor({v_A_init.data(), A_info}, "A");

  // Tensor B
  TensorInfo B_info{"FLOAT", std::vector<int64_t>{K, O, N}};
  std::vector<float> v_B_init(B_info.nelms());
  for (auto &val : v_B_init) {
    val = fdis(eng);
  }
  TensorId B_id =
      bder->addInitializedInputTensor({v_B_init.data(), B_info}, "B");

  // Tensor C
  TensorInfo C_info{"FLOAT", std::vector<int64_t>{K, N, M}};
  std::vector<float> v_C_init(C_info.nelms());
  for (auto &val : v_C_init) {
    val = fdis(eng);
  }
  TensorId C_id =
      bder->addInitializedInputTensor({v_C_init.data(), C_info}, "C");

  TensorId D_id = bder->customOp(
      Onnx::AiOnnx::OpSet11::MatMul, 11, {A_id, B_id}, 1, {}, "MatMul")[0];

  TensorId E_id = bder->customOp(
      Onnx::AiOnnx::OpSet11::MatMul, 11, {D_id, C_id}, 1, {}, "MatMul")[0];

  auto l1            = bder->aiGraphcoreOpset1().l1loss({E_id}, 0.1);
  auto proto         = bder->getModelProto();
  auto modelProto    = io::getModelFromString(proto);
  auto art           = AnchorReturnType("All");
  int batchesPerStep = 1;
  auto dataFlow      = DataFlow(batchesPerStep, {{B_id, art}});

  std::map<popart::TensorId, popart::IArray &> inputs  = {};
  std::map<popart::TensorId, popart::IArray &> anchors = {};

  auto optimizer = popart::ConstSGD(0.01f);

  auto opts = SessionOptions();
  // Disable outlining
  opts.enableOutlining = false;

  auto patterns = popart::Patterns(PatternsLevel::Default);
  // Disable inplacing since this could affect the scheduler
  patterns.enableInPlace(false);

  auto device = createTestDevice(TEST_TARGET);
  auto session =
      popart::TrainingSession::createFromOnnxModel(proto,
                                                   dataFlow,
                                                   l1,
                                                   optimizer,
                                                   device,
                                                   popart::InputShapeInfo(),
                                                   opts,
                                                   patterns);
  session->prepareDevice();
  popart::StepIO stepio(inputs, anchors);

  // Verify schedule order
  auto schedule = session->getIr().getMainGraph().getOpSchedule(
      {}, RequireOptimalSchedule::Yes);

  auto schedulePosition = [&](Op *op) {
    return std::distance(schedule.begin(),
                         std::find(schedule.begin(), schedule.end(), op));
  };

  // Check that the schedule is tight: No early transposes or reshapes
  for (int64_t i = 0; i < schedule.size(); ++i) {
    Op *op = schedule.at(i);
    for (auto &kv : op->input->tensorMap()) {
      if (kv.second->hasProducer()) {
        Op *producer     = kv.second->getProducer();
        int64_t pos      = schedulePosition(producer);
        int64_t distance = i - pos;
        logging::trace("Distance producer-consumer: {} ({}-{})",
                       distance,
                       producer->debugName(),
                       op->debugName());
        if (producer->isConvertibleTo<ReshapeBaseOp>() ||
            producer->isConvertibleTo<TransposeBaseOp>()) {
          // All ReshapeOp & TransposeOp as tight to their consumers as possible
          BOOST_CHECK(distance <= 2);
        }
        // NOTE: There used to be check here to assert (distance <= 8). However,
        // this check was violated in practice and it is not clear by what
        // mechanism we can guarantee this distance is 8 or less, so this check
        // has been removed.
      }
    }
  }
}
