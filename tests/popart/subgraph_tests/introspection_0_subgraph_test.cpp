// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE Introspection0SubgraphTest

#include <boost/test/unit_test.hpp>
#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/filereader.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/call.hpp>
#include <popart/op/ipucopy.hpp>
#include <popart/op/l1.hpp>
#include <popart/op/nll.hpp>
#include <popart/optimizer.hpp>
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

  auto loss0 =
      std::make_shared<L1Loss>(out[0], "l1LossVal_0", 0.1, ReductionType::Mean);
  loss0->virtualGraph(0);
  auto loss1 =
      std::make_shared<L1Loss>(out[4], "l1LossVal_1", 0.1, ReductionType::Mean);
  loss1->virtualGraph(1);

  auto device = createTestDevice(TEST_TARGET, nIpus);

  Ir ir;
  ir.prepare({modelProto,
              InputShapeInfo(),
              dataFlow,
              {loss0, loss1},
              &optimizer,
              *device,
              userOptions,
              Patterns(PatternsLevel::NoPatterns)});

  auto sched = ir.getMainGraph().getOpSchedule({});

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
        BOOST_CHECK(op->getIntrospectionInVirtualGraphId(index) == 0);
      }

      if (tensor->id == "w1" || tensor->id == "input1") {
        BOOST_CHECK(tensor->getVirtualGraphId() == 1);
        auto index = op->input->indicesMap().at(tensor)[0];
        BOOST_CHECK(op->getIntrospectionInVirtualGraphId(index) == 1);
      }
    }
  }

  auto &tensors = ir.getMainGraphTensors();
  auto ids      = tensors.getAllTensorIds();
  logging::debug("Tensor IDs: {}", ids);
}
