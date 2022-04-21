// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE Introspection0SubgraphTest

#include <boost/test/unit_test.hpp>
#include <filereader.hpp>
#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/call.hpp>
#include <popart/op/identity.hpp>
#include <popart/op/ipucopy.hpp>
#include <popart/sgd.hpp>
#include <popart/tensor.hpp>
#include <popart/tensordata.hpp>
#include <popart/tensors.hpp>
#include <popart/testdevice.hpp>

BOOST_AUTO_TEST_CASE(Introspection0_Subgraph) {

  using namespace popart;

  auto builder     = Builder::create();
  auto aiOnnx      = builder->aiOnnxOpset9();
  auto aiGraphcore = builder->aiGraphcoreOpset1();
  TensorInfo info{"FLOAT", std::vector<int64_t>{4, 6}};
  std::vector<float> wVals(4 * 6, 1.0f);
  ConstVoidData wData0 = {wVals.data(), info};
  ConstVoidData wData1 = {wVals.data(), info};

  std::vector<TensorId> input = {builder->addInputTensor(info, "input0"),
                                 builder->addInputTensor(info, "input1")};
  std::vector<TensorId> w = {builder->addInitializedInputTensor(wData0, "w0"),
                             builder->addInitializedInputTensor(wData1, "w1")};

  constexpr int64_t nIpus{2};

  std::vector<TensorId> out(8);

  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      for (int ipu = 0; ipu < 2; ++ipu) {
        out[ipu * 4 + i * 2 + j] = aiOnnx.add({input[ipu], w[ipu]}, "add");
        builder->virtualGraph(out[ipu * 4 + i * 2 + j], ipu);
      }
    }
    for (int ipu = 0; ipu < 2; ++ipu) {
      out[ipu * 4 + i * 2] = aiOnnx.add(
          {out[ipu * 4 + i * 2 + 0], out[ipu * 4 + i * 2 + 1]}, "add");
      builder->virtualGraph(out[ipu * 4 + i * 2], ipu);
    }
  }
  for (int ipu = 0; ipu < 2; ++ipu) {
    out[ipu * 4] = aiOnnx.concat({out[ipu * 4 + 0], out[ipu * 4 + 2]}, 0);
    builder->virtualGraph(out[ipu * 4], ipu);
  }

  auto finalSum = aiOnnx.sum({out[0], out[4]});
  builder->virtualGraph(finalSum, 1);
  auto finalLoss = aiGraphcore.l1loss({finalSum}, 0.1);
  builder->virtualGraph(finalLoss, 1);

  auto proto      = builder->getModelProto();
  auto modelProto = io::getModelFromString(proto);
  auto dataFlow   = DataFlow(
      100,
      {{out[0], AnchorReturnType("All")}, {out[4], AnchorReturnType("All")}});

  SessionOptions userOptions;
  userOptions.virtualGraphMode               = VirtualGraphMode::Manual;
  userOptions.autoRecomputation              = RecomputationType::None;
  userOptions.enableOutlining                = true;
  userOptions.outlineThreshold               = -100.f;
  userOptions.enableOutliningCopyCostPruning = false;

  std::map<std::string, std::string> deviceOpts{
      {"numIPUs", std::to_string(nIpus)}};

  auto optimizer = ConstSGD(0.01);

  auto device = createTestDevice(TEST_TARGET, nIpus);

  Ir ir;
  ir.prepare({modelProto,
              InputShapeInfo(),
              dataFlow,
              finalLoss,
              &optimizer,
              *device,
              userOptions,
              Patterns(PatternsLevel::NoPatterns).enableRuntimeAsserts(false)});

  auto sched = ir.getMainGraph().getOpSchedule({}, RequireOptimalSchedule::Yes);

  for (int i = 0; i < sched.size(); ++i) {
    auto op = sched[i];
    logging::trace("OP: {}, ID: {}, value: {}, priority: {}",
                   op->debugName(),
                   op->getSubgraphEquivId(),
                   op->getSubgraphValue(),
                   op->settings.schedulePriority);
  }

  for (int i = 0; i < sched.size(); ++i) {
    auto op = sched[i];
    for (auto tensor : op->input->tensors()) {
      logging::debug(
          "Tensor: {}, VGID: {}", tensor->id, tensor->getVirtualGraphId());

      if (tensor->id == "w0" || tensor->id == "input0") {
        BOOST_CHECK(tensor->getVirtualGraphId() == 0);
        auto index = op->input->indicesMap().at(tensor)[0];
        BOOST_CHECK(op->getIntrospectionInVirtualGraphId(index).first == 0);
      }

      if (tensor->id == "w1" || tensor->id == "input1") {
        BOOST_CHECK(tensor->getVirtualGraphId() == 1);
        auto index = op->input->indicesMap().at(tensor)[0];
        BOOST_CHECK(op->getIntrospectionInVirtualGraphId(index).first == 1);
      }
    }
  }

  auto &tensors = ir.getMainGraphTensors();
  auto ids      = tensors.getAllTensorIds();
  logging::debug("Tensor IDs: {}", ids);
}
