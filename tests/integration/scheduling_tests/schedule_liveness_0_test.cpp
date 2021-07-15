// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE ScheduleLivness0Test

#include <boost/test/unit_test.hpp>
#include <filereader.hpp>
#include <vector>
#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/names.hpp>
#include <popart/op/add.hpp>
#include <popart/op/l1.hpp>
#include <popart/op/scale.hpp>
#include <popart/optimizer.hpp>
#include <popart/sessionoptions.hpp>
#include <popart/tensordata.hpp>
#include <popart/testdevice.hpp>
#include <popart/topocons.hpp>

using namespace popart;

// Graph
//
//
// in ------ scale ----
//       |             |
//       |             |
//       |             |
//       |-- scale -- add
//       |             |
//       |             |
//       |             |
//       |-- scale -- add
//       |             |
//       |             |
//       |             |
//       |-- scale -- add
//       |             |
//       |             |
//       |             |
//       |-- scale -- add
//       |             |
//       |             |
//       |             |
//       |-- scale -- add
//       |             |
//       |             |
//       |             |
//       |-- scale -- add
//       |             |
//       |             |
//       |             |
//       |-- scale -- add
//       |             |
//       |             |
//       |             |
//       |-- scale -- add ------- output
//
// check that schedule is:
// scale, scale, add, scale, add, scale, add, scale, add, scale, add etc.
// i.e. adds scheduled as early as possible.

void setIr(uint64_t N, Ir &ir) {

  // Build ONNX model (same as ScheduleLiveness0Test)
  auto builder     = Builder::create();
  auto aiOnnx      = builder->aiOnnxOpset9();
  auto aiGraphcore = builder->aiGraphcoreOpset1();
  TensorInfo input_shape{"FLOAT", std::vector<int64_t>{32, 32}};
  auto in0 = builder->addInputTensor(input_shape);
  std::vector<TensorId> scaledTensors;
  for (int i = 0; i < N; ++i) {
    scaledTensors.push_back(aiGraphcore.scale({in0}, i + 0.));
  }
  auto sum = aiOnnx.add({scaledTensors[0], scaledTensors[1]});
  for (uint64_t i = 2; i < N; ++i) {
    sum = aiOnnx.add({sum, scaledTensors[i]});
  }
  auto proto      = builder->getModelProto();
  auto modelProto = io::getModelFromString(proto);
  auto dataFlow   = DataFlow(1, {{sum, AnchorReturnType("All")}});
  auto device     = createTestDevice(TEST_TARGET);
  SessionOptions opts;
  opts.autoRecomputation              = RecomputationType::None;
  opts.enableOutlining                = false;
  opts.enableOutliningCopyCostPruning = false;
  opts.mergeVarUpdate                 = MergeVarUpdateType::None;

  ir.prepare({modelProto,
              InputShapeInfo(),
              dataFlow,
              {},
              nullptr,
              *device,
              opts,
              Patterns({}).enableRuntimeAsserts(false)});
}

BOOST_AUTO_TEST_CASE(ScheduleLiveness0Test) {

  uint64_t N = 50;
  Ir ir;
  setIr(N, ir);

  auto opSchedule = ir.getOpSchedule({}, RequireOptimalSchedule::Yes);
  BOOST_CHECK(dynamic_cast<ScaleOp *>(opSchedule[0]));
  BOOST_CHECK(dynamic_cast<ScaleOp *>(opSchedule[1]));
  for (uint64_t i = 2; i < opSchedule.size(); ++i) {
    if (i % 2 == 0) {
      BOOST_CHECK(dynamic_cast<AddOp *>(opSchedule[i]));
    } else {
      BOOST_CHECK(dynamic_cast<ScaleOp *>(opSchedule[i]));
    }
  }
}

BOOST_AUTO_TEST_CASE(ScheduleLiveness1Test) {

  uint64_t N = 50;
  Ir ir;
  setIr(N, ir);

  // set priorities, manually, for certain Ops. In particular, every fifth scale
  // op is given high priority.
  auto scaleOps = ir.opsOfType(Onnx::CustomOperators::Scale_1);
  for (auto op : scaleOps) {
    auto scOp      = dynamic_cast<ScaleOp *>(op);
    int scaleAsInt = static_cast<int>(scOp->getScaleFactor());
    if (scaleAsInt % 5 == 0) {
      scOp->settings.schedulePriority = scOp->getScaleFactor() + 100.;
    }
  }

  // expectation: adds as early as possible, except when scales have priority.
  std::vector<std::pair<bool, double>> expected;
  for (int64_t i = N - 1; i >= 0; --i) {
    if (i % 5 == 0) {
      expected.push_back({true, i});
    }
  }
  for (uint64_t i = 1; i < N; ++i) {
    if (i % 5 != 0) {
      expected.emplace_back(true, static_cast<double>(i));
    }
    expected.emplace_back(false, -1);
  }

  auto opSchedule = ir.getOpSchedule({}, RequireOptimalSchedule::Yes);

  if (opSchedule.size() != expected.size()) {
    throw error("sizes do not match");
  }
  for (uint64_t i = 0; i < opSchedule.size(); ++i) {
    auto expectScale = expected[i].first;
    auto scaleFactor = expected[i].second;
    auto scaleOp     = dynamic_cast<ScaleOp *>(opSchedule[i]);

    if (expectScale) {
      BOOST_CHECK(scaleOp);
      BOOST_CHECK(scaleOp->getScaleFactor() == scaleFactor);
    } else {
      BOOST_CHECK(!scaleOp);
    }
  }
}

BOOST_AUTO_TEST_CASE(ScheduleLiveness2Test) {

  uint64_t N = 5;
  Ir ir;
  setIr(N, ir);

  // set priorities, manually, for certain Ops. In particular, every fifth scale
  // op is given high priority.
  // In addition, define a tied topo con
  auto scaleOps = ir.opsOfType(Onnx::CustomOperators::Scale_1);
  for (auto op : scaleOps) {
    auto scOp                       = dynamic_cast<ScaleOp *>(op);
    int scaleAsInt                  = static_cast<int>(scOp->getScaleFactor());
    scOp->settings.schedulePriority = 3.0 - scOp->getScaleFactor();
  }

  // This topo con should not disturb the priority order
  ir.getMainGraph().topoCons->insert(scaleOps[2], scaleOps[4], true);

  auto opSchedule = ir.getOpSchedule({}, RequireOptimalSchedule::Yes);

  // Wrong schedule if lex order of tied topo cons is -1 and priorities is -1:
  // Priority, Op
  // 3 Op(ai.graphcore.Scale:1, inputs=[input], outputs=[Scale:0])
  // 2 Op(ai.graphcore.Scale:1, inputs=[input], outputs=[Scale:0/1])
  // 0 Op(ai.onnx.Add:7, inputs=[Scale:0, Scale:0/1], outputs=[Add:0])
  // 1 Op(ai.graphcore.Scale:1, inputs=[input], outputs=[Scale:0/2])
  // 0 Op(ai.onnx.Add:7, inputs=[Add:0, Scale:0/2], outputs=[Add:0/1])
  // 0 Op(ai.graphcore.Scale:1, inputs=[input], outputs=[Scale:0/3])
  // 0 Op(ai.onnx.Add:7, inputs=[Add:0/1, Scale:0/3], outputs=[Add:0/2])
  // -1 Op(ai.graphcore.Scale:1, inputs=[input], outputs=[Scale:0/4])
  // 0 Op(ai.onnx.Add:7, inputs=[Add:0/2, Scale:0/4], outputs=[Add:0/3])

  // Correct schedule if lex order of tied topo cons is -1 and priorities is -2:
  // Priority, Op
  // 3 Op(ai.graphcore.Scale:1, inputs=[input], outputs=[Scale:0])
  // 2 Op(ai.graphcore.Scale:1, inputs=[input], outputs=[Scale:0/1])
  // 1 Op(ai.graphcore.Scale:1, inputs=[input], outputs=[Scale:0/2])
  // 0 Op(ai.onnx.Add:7, inputs=[Scale:0, Scale:0/1], outputs=[Add:0])
  // 0 Op(ai.onnx.Add:7, inputs=[Add:0, Scale:0/2], outputs=[Add:0/1])
  // 0 Op(ai.graphcore.Scale:1, inputs=[input], outputs=[Scale:0/3])
  // 0 Op(ai.onnx.Add:7, inputs=[Add:0/1, Scale:0/3], outputs=[Add:0/2])
  // -1 Op(ai.graphcore.Scale:1, inputs=[input], outputs=[Scale:0/4])
  // 0 Op(ai.onnx.Add:7, inputs=[Add:0/2, Scale:0/4], outputs=[Add:0/3])

  // Check schedule order
  for (uint64_t i = 0; i < opSchedule.size(); ++i) {
    auto scaleOp = dynamic_cast<ScaleOp *>(opSchedule[i]);
    if (i < 3) {
      BOOST_CHECK(opSchedule[i]->settings.schedulePriority == 3.0 - i);
      BOOST_CHECK(scaleOp);
    }
    if (i == opSchedule.size() - 2) {
      BOOST_CHECK(opSchedule[i]->settings.schedulePriority == -1);
      BOOST_CHECK(scaleOp);
    }
  }
}
