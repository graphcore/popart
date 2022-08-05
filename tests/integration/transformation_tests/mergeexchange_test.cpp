// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE MergeExchangeTest

#include <boost/test/unit_test.hpp>
#include <map>
#include <memory>
#include <string>
#include <testdevice.hpp>
#include <utility>
#include <vector>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/exchange/remote.hpp>
#include <popart/op/init.hpp>
#include <popart/session.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/topocons.hpp>
#include <popart/transforms/mergeexchange.hpp>
#include <popart/transforms/remotesetup.hpp>

#include "popart/datatype.hpp"
#include "popart/graphcoreoperators.hpp"
#include "popart/graphid.hpp"
#include "popart/logging.hpp"
#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/popx/exchangebundle.hpp"
#include "popart/popx/irlowering.hpp"
#include "popart/sessionoptions.hpp"
#include "popart/stepio.hpp"
#include "popart/tensor.hpp"
#include "popart/tensordebuginfo.hpp"

namespace popart {
class IArray;
} // namespace popart

using namespace popart;

namespace {

/**
 * Create a model that includes two RemoteLoads and two RemoteStores which have
 * data dependencies on each other
 * \param ir             IR to create the model in
 * \param mergeExchange  True if the \a MergeExchange transform should be run
 */
void createModelWithRemoteExchange0(Ir *ir, bool mergeExchange) {
  // Construct IR and main graph.
  Graph &g = ir->getMainGraph();

  const TensorInfo tInfo{DataType::FLOAT, Shape{2, 2}};

  auto initOp0 = g.createConnectedOp<InitOp>({},
                                             {{InitOp::getOutIndex(), "init0"}},
                                             Onnx::CustomOperators::Init_1,
                                             tInfo,
                                             TensorType::ActGrad,
                                             InitType::Zero,
                                             Op::Settings{g, "init"});
  auto initOp1 = g.createConnectedOp<InitOp>({},
                                             {{InitOp::getOutIndex(), "init1"}},
                                             Onnx::CustomOperators::Init_1,
                                             tInfo,
                                             TensorType::ActGrad,
                                             InitType::Zero,
                                             Op::Settings{g, "init"});
  auto loadOp0 = g.createConnectedOp<RemoteLoadInplaceOp>(
      {{0, "init0"}},
      {{0, "load0"}},
      Onnx::CustomOperators::RemoteLoadInplace,
      Op::Settings{g, "load"},
      0);
  auto loadOp1 = g.createConnectedOp<RemoteLoadInplaceOp>(
      {{0, "init1"}},
      {{0, "load1"}},
      Onnx::CustomOperators::RemoteLoadInplace,
      Op::Settings{g, "load"},
      1);

  auto storeOp0 =
      g.createConnectedOp<RemoteStoreOp>({{0, "load0"}},
                                         {},
                                         Onnx::CustomOperators::RemoteStore,
                                         Op::Settings{g, "store"},
                                         1);
  auto storeOp1 =
      g.createConnectedOp<RemoteStoreOp>({{0, "load1"}},
                                         {},
                                         Onnx::CustomOperators::RemoteStore,
                                         Op::Settings{g, "store"},
                                         0);

  g.topoCons->insert(initOp0, loadOp1, false);
  g.topoCons->insert(initOp1, loadOp0, false);
  g.topoCons->insert(loadOp0, storeOp1, false);
  g.topoCons->insert(loadOp1, storeOp0, false);

  g.topoCons->insert(initOp0, initOp1, false);
  g.topoCons->insert(loadOp0, loadOp1, false);
  g.topoCons->insert(storeOp0, storeOp1, false);

  auto &opts                   = ir->getSessionOptions();
  opts.enableExplicitMainLoops = true;
  opts.useHostCopyOps          = true;

  if (mergeExchange) {
    ir->applyTransform(MergeExchange::id(), g);
  }
  ir->applyTransform(RemoteSetup::id(), g);
  ir->updateVertices();
  ir->setIsPrepared();
}

/**
 * Create a model that mimics phased execution, loading and storing
 * from the same remote buffers in a merged RemoteExchange
 * \param ir             IR to create the model in
 * \param numBuffers     Number of remote buffers to use
 * \param mergeExchange  True if the \a MergeExchange transform should be run
 */
void createModelWithRemoteExchange1(Ir *ir,
                                    int numBuffers,
                                    bool mergeExchange) {
  // Construct IR and main graph.
  Graph &g = ir->getMainGraph();

  const TensorInfo tInfo{DataType::FLOAT, Shape{2, 2}};

  std::vector<InitOp *> initOps;
  initOps.reserve(numBuffers * 2);

  // Add one InitOp per RemoteStore and RemoteLoad (2 * numBuffers in total)
  for (int bufferId = 0; bufferId < numBuffers; ++bufferId) {
    for (int initId = 0; initId < 2; ++initId) {
      initOps.push_back(g.createConnectedOp<InitOp>(
          {},
          {{InitOp::getOutIndex(),
            logging::format("init_{}_{}", bufferId, initId)}},
          Onnx::CustomOperators::Init_1,
          tInfo,
          TensorType::ActGrad,
          InitType::Zero,
          Op::Settings{g, "init"}));
    }
  }

  for (int bufferId = 0; bufferId < numBuffers; ++bufferId) {
    // Connect one RemoteLoadOp per remote buffer
    auto load = g.createConnectedOp<RemoteLoadOp>(
        {{0, logging::format("init_{}_{}", bufferId, 0)}},
        {{0, logging::format("load_{}", bufferId)}},
        Onnx::CustomOperators::RemoteLoad,
        Op::Settings{g, "load"},
        bufferId);

    load->pruneable = false;

    // Connect one RemoteStoreOp per remote buffer with no sequential data
    // dependency to the RemoteLoadOp, such that they can be merged to
    // a single MultiExchangeOp
    auto store = g.createConnectedOp<RemoteStoreOp>(
        {{0, logging::format("init_{}_{}", bufferId, 0)}},
        {},
        Onnx::CustomOperators::RemoteStore,
        Op::Settings{g, "store"},
        bufferId);

    // Add topocons such that all InitOps occur before all RemoteLoad and
    // RemoteStore operations
    for (auto init : initOps) {
      g.topoCons->insert(init, load, false);
      g.topoCons->insert(init, store, false);
    }

    store->pruneable = false;
  }

  auto &opts                   = ir->getSessionOptions();
  opts.enableExplicitMainLoops = true;
  opts.useHostCopyOps          = true;

  if (mergeExchange) {
    ir->applyTransform(MergeExchange::id(), g);
  }
  ir->applyTransform(RemoteSetup::id(), g);
  ir->updateVertices();
  ir->setIsPrepared();
}

} // namespace

// Testing that adjacent RemoteLoad/RemoteStore are merged if they have
// no data dependencies, and not merged if they do.
// Compares that unmerged and merged remote exchanges yield the same result.
BOOST_AUTO_TEST_CASE(TestMergeRemoteExchange) {

  std::vector<float> in0{0, 1, 2, 3};
  std::vector<float> in1{7, 6, 5, 4};
  std::vector<float> run0_out0(4, -1);
  std::vector<float> run0_out1(4, -1);
  std::vector<float> run1_out0(4, -1);
  std::vector<float> run1_out1(4, -1);

  auto run = [&in0, &in1](bool mergeExchange,
                          std::vector<float> &out0,
                          std::vector<float> &out1) {
    auto ir = std::make_unique<Ir>();

    createModelWithRemoteExchange0(ir.get(), mergeExchange);

    if (mergeExchange) {
      BOOST_REQUIRE_EQUAL(
          ir->opsOfType(Onnx::CustomOperators::RemoteLoadInplace).size(), 0);
      BOOST_REQUIRE_EQUAL(
          ir->opsOfType(Onnx::CustomOperators::RemoteStore).size(), 0);
      BOOST_REQUIRE_EQUAL(
          ir->opsOfType(Onnx::CustomOperators::MultiExchange).size(), 2);
    } else {
      BOOST_REQUIRE_EQUAL(
          ir->opsOfType(Onnx::CustomOperators::RemoteLoadInplace).size(), 2);
      BOOST_REQUIRE_EQUAL(
          ir->opsOfType(Onnx::CustomOperators::RemoteStore).size(), 2);
      BOOST_REQUIRE_EQUAL(
          ir->opsOfType(Onnx::CustomOperators::MultiExchange).size(), 0);
    }

    const auto session = TrainingSession::createFromIr(
        std::move(ir), createTestDevice(TEST_TARGET));
    session->prepareDevice();

    // Populate remote buffers
    session->copyToRemoteBuffer(static_cast<void *>(in0.data()), "RB_0", 0, 0);
    session->copyToRemoteBuffer(static_cast<void *>(in1.data()), "RB_1", 0, 0);

    // Run the model
    StepIO stepio({}, {});
    session->weightsFromHost();
    session->run(stepio);

    // Read remote buffers
    session->copyFromRemoteBuffer(
        "RB_0", static_cast<void *>(out0.data()), 0, 0);
    session->copyFromRemoteBuffer(
        "RB_1", static_cast<void *>(out1.data()), 0, 0);
  };

  run(false, run0_out0, run0_out1);
  run(true, run1_out0, run1_out1);

  BOOST_REQUIRE_EQUAL(in0, run0_out1);
  BOOST_REQUIRE_EQUAL(in1, run0_out0);
  BOOST_REQUIRE_EQUAL(run0_out0, run1_out0);
  BOOST_REQUIRE_EQUAL(run0_out1, run1_out1);
}

// Testing that adjacent RemoteLoad/RemoteStore are merged even if they use the
// same remote buffer, and that this triggers using two landing pad tensors
// per remote buffer instead of one.
BOOST_AUTO_TEST_CASE(TestMergeRemoteExchangeLandingPads) {

  auto numBuffers = 3;

  auto run = [&numBuffers](bool mergeExchange) {
    auto ir = std::make_unique<Ir>();

    createModelWithRemoteExchange1(ir.get(), numBuffers, mergeExchange);

    if (mergeExchange) {
      BOOST_REQUIRE_EQUAL(
          ir->opsOfType(Onnx::CustomOperators::RemoteLoadInplace).size(), 0);
      BOOST_REQUIRE_EQUAL(
          ir->opsOfType(Onnx::CustomOperators::RemoteStore).size(), 0);
      BOOST_REQUIRE_EQUAL(
          ir->opsOfType(Onnx::CustomOperators::MultiExchange).size(), 1);

      const auto session = TrainingSession::createFromIr(
          std::move(ir), createTestDevice(TEST_TARGET));
      session->prepareDevice();

      for (int i = 0; i < numBuffers; ++i) {
        BOOST_REQUIRE_EQUAL(
            session->getIrLowering()
                .getExchangeBundle()
                .getRemoteBufferSeparateLoadStorePadsRequired(i),
            true);
        BOOST_REQUIRE_EQUAL(session->getIrLowering()
                                .getExchangeBundle()
                                .getRemoteBuffer(i)
                                .second.size(),
                            2);
      }

    } else {
      BOOST_REQUIRE_EQUAL(
          ir->opsOfType(Onnx::CustomOperators::RemoteLoad).size(), numBuffers);
      BOOST_REQUIRE_EQUAL(
          ir->opsOfType(Onnx::CustomOperators::RemoteStore).size(), numBuffers);
      BOOST_REQUIRE_EQUAL(
          ir->opsOfType(Onnx::CustomOperators::MultiExchange).size(), 0);

      const auto session = TrainingSession::createFromIr(
          std::move(ir), createTestDevice(TEST_TARGET));
      session->prepareDevice();

      for (int i = 0; i < numBuffers; ++i) {
        BOOST_REQUIRE_EQUAL(
            session->getIrLowering()
                .getExchangeBundle()
                .getRemoteBufferSeparateLoadStorePadsRequired(i),
            false);
        BOOST_REQUIRE_EQUAL(session->getIrLowering()
                                .getExchangeBundle()
                                .getRemoteBuffer(i)
                                .second.size(),
                            1);
      }
    }
  };

  run(false);
  run(true);
}
