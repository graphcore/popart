// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE GraphOutput0InplaceTest

#include <boost/test/unit_test.hpp>
#include <cstddef>
#include <cstdint>
#include <filereader.hpp>
#include <memory>
#include <string>
#include <testdevice.hpp>
#include <vector>
#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/graph.hpp>
#include <popart/inputshapeinfo.hpp>
#include <popart/ir.hpp>
#include <popart/tensorinfo.hpp>

#include "popart/builder.gen.hpp"
#include "popart/error.hpp"
#include "popart/graphid.hpp"
#include "popart/logging.hpp"
#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/operators.hpp"
#include "popart/patterns/patterns.hpp"
#include "popart/scheduler_requireoptimal.hpp"
#include "popart/voiddata.hpp"

using namespace popart;

BOOST_AUTO_TEST_CASE(InplaceLoopTest0) {
  //
  //  in0  in1
  //   |    |
  //   |  Reshape     (inplaced, priority 200)
  //   |    |
  //   |  Scale       (not inplaced, priority 100)
  //   |    |         (would modify implicit input indirectly)
  //   |  Reshape     (inplaced, priority 200)
  //   |  /
  //  Add             (inplaced - LHS on explicit input)
  //   |
  //  out0

  auto superBuilder     = Builder::create();
  auto builder          = &(superBuilder->createSubgraphBuilder());
  auto aiOnnx           = builder->aiOnnxOpset11();
  auto aiGraphcore      = builder->aiGraphcoreOpset1();
  auto superAiOnnx      = superBuilder->aiOnnxOpset11();
  auto superAiGraphcore = superBuilder->aiGraphcoreOpset1();

  TensorInfo shape0{"FLOAT", std::vector<int64_t>{1, 1}};
  TensorInfo shape1{"FLOAT", std::vector<int64_t>{1}};
  TensorInfo const_bool_shape{"BOOL", std::vector<int64_t>{}};
  TensorInfo const_int32_shape{"INT32", std::vector<int64_t>{}};
  TensorInfo const_int64_shape{"INT64", std::vector<int64_t>{}};

  // Explicit required inputs
  auto iter      = builder->addInputTensor(const_int64_shape);
  auto keepgoing = builder->addInputTensor(const_bool_shape);

  // Explicit input
  auto superIn0 = superBuilder->addInputTensor(shape0);
  auto in0      = builder->addUntypedInputTensor(superIn0);

  // Implicit input
  auto superIn1 = superBuilder->addInputTensor(shape0);

  int const_int_data[1] = {10};
  popart::ConstVoidData const_int_cvdata{const_int_data, const_int32_shape};
  auto M = superAiOnnx.constant(const_int_cvdata);

  bool const_bool_data[1] = {true};
  popart::ConstVoidData const_bool_cvdata{const_bool_data, const_bool_shape};
  auto cond = superAiOnnx.constant(const_bool_cvdata);

  auto shape0_t = aiOnnx.constant({shape0.shape().data(), {"INT64", Shape{2}}});
  auto shape1_t = aiOnnx.constant({shape1.shape().data(), {"INT64", Shape{1}}});

  // Can inplace
  auto re0 = aiOnnx.reshape({superIn1, shape1_t});
  builder->setInplacePreferences(re0, {{"ReshapeInplace", 200}});

  // Can not inplace
  auto sc = aiGraphcore.scale({re0}, 1.5, "sc0");
  builder->setInplacePreferences(sc, {{"ScaleInplace", 100}});

  // Can inplace
  auto re1 = aiOnnx.reshape({sc, shape0_t});
  builder->setInplacePreferences(re1, {{"ReshapeInplace", 200}});

  auto a = aiOnnx.add({in0, re1});

  builder->addOutputTensor(keepgoing);
  builder->addOutputTensor(a);

  auto acts = superAiOnnx.loop({M, cond, superIn0}, 1, *builder);

  auto proto      = superBuilder->getModelProto();
  auto modelProto = io::getModelFromString(proto);

  // Create the IR
  auto dataFlow = DataFlow(1, {{acts[0], AnchorReturnType("All")}});
  auto device   = createTestDevice(TEST_TARGET);

  Ir ir;
  ir.prepare({modelProto,
              InputShapeInfo(),
              dataFlow,
              {},
              nullptr,
              *device,
              {},
              Patterns(PatternsLevel::NoPatterns)
                  .enableRuntimeAsserts(false)
                  .enableInPlace(true)});

  auto graphSched = ir.getGraphSchedule();
  BOOST_CHECK(graphSched.size() == 2);
  for (auto g : graphSched) {
    auto gSched = g->getOpSchedule({}, RequireOptimalSchedule::Yes);

    for (size_t i = 0; i < gSched.size(); ++i) {
      logging::trace("{} - {} - {}", g->id.str(), i, gSched.at(i)->debugName());
    }

    if (gSched.size() == 1) {
      BOOST_CHECK_EQUAL(gSched[0]->opid, Onnx::AiOnnx::OpSet11::Loop);
    } else if (gSched.size() == 4) {
      BOOST_CHECK_EQUAL(gSched[0]->opid, Onnx::CustomOperators::ReshapeInplace);
      BOOST_CHECK_EQUAL(gSched[1]->opid, Onnx::CustomOperators::Scale_1);
      BOOST_CHECK_EQUAL(gSched[2]->opid, Onnx::CustomOperators::ReshapeInplace);
      BOOST_CHECK_EQUAL(gSched[3]->opid, Onnx::CustomOperators::AddLhsInplace);
    } else {
      throw error(
          "Expected main schedule of size 1, loop of size 4. Observed size {}",
          gSched.size());
    }
  }
}
