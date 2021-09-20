// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE DirectViewChangeTest

#include <memory>
#include <vector>

#include <boost/test/unit_test.hpp>

#include <filereader.hpp>
#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/graph.hpp>
#include <popart/graphtransformer.hpp>
#include <popart/ir.hpp>
#include <popart/names.hpp>
#include <popart/op/identity.hpp>
#include <popart/op/l1.hpp>
#include <popart/sgd.hpp>
#include <popart/tensor.hpp>
#include <popart/tensordata.hpp>
#include <popart/tensors.hpp>
#include <popart/testdevice.hpp>
#include <popart/util.hpp>

using namespace popart;

/**
    ┌───Input────┐
    │            │
    ▼            ▼
 reshape       sqrt
    │            │
    ▼            ▼
 identity    reshape
    │            │
    ▼            ▼
transpose    identity
    │            │
    │            ▼
    │        transpose
    │            ┼
   t0            t1
    ──►  Add ◄────
          │
          │
          ▼
        L1Loss

  Where t0 and t1 are anchored to avoid them getting removed.


Inplacing is on, and everything should get inplaced apart from the sqrt.

getDirectViewChain(Input, t0) should be {IdentityInplace, TransposeInplace,
ReshapeInplace} (last op first)

getDirectViewChain(Input, t1) should be {} due to the the sqrt
blocking.
 **/
BOOST_AUTO_TEST_CASE(DirectViewChangeTest0) {

  Shape inShape  = {2, 5, 3, 4};
  Shape outShape = {10, 12};

  Shape outShapeSize = {static_cast<int64_t>(outShape.size())};
  TensorInfo inInfo{"FLOAT", inShape};
  ConstVoidData outShapeData = {outShape.data(), {"INT64", outShapeSize}};

  // Build an onnx model
  auto builder     = Builder::create();
  auto aiOnnx      = builder->aiOnnxOpset9();
  auto aiGraphcore = builder->aiGraphcoreOpset1();
  auto newShapeId  = aiOnnx.constant(outShapeData, "outShapeData");
  auto inId        = builder->addInputTensor(inInfo);

  // path 1
  auto outId = aiOnnx.identity({inId});
  outId      = aiOnnx.reshape({outId, newShapeId});
  outId      = aiOnnx.transpose({outId}, {1, 0});
  outId      = aiOnnx.identity({outId});

  // path 2
  auto outId2 = aiOnnx.sqrt({inId});
  outId2      = aiOnnx.reshape({outId2, newShapeId});
  outId2      = aiOnnx.identity({outId2});
  outId2      = aiOnnx.transpose({outId2}, {1, 0});

  auto addId = aiOnnx.add({outId, outId2});

  auto lossId = aiGraphcore.l1loss({addId}, 0.1);

  auto proto      = builder->getModelProto();
  auto modelProto = io::getModelFromString(proto);

  // Create the IR, adding outId as an anchor
  auto art       = AnchorReturnType("All");
  auto dataFlow  = DataFlow(1, {{outId, art}, {outId2, art}});
  auto optimizer = ConstSGD(0.01);
  auto device    = createTestDevice(TEST_TARGET);

  auto patterns = Patterns::create({"PostNRepl"}).enableRuntimeAsserts(false);
  patterns.enableInPlace(true);

  SessionOptions opts;

  Ir ir;
  ir.prepare({modelProto,
              InputShapeInfo(),
              dataFlow,
              lossId,
              &optimizer,
              *device,
              opts,
              patterns});

  auto &graph = ir.getMainGraph();
  auto inT    = graph.getTensors().get(inId);
  auto outT   = graph.getTensors().get(outId);
  auto outT2  = graph.getTensors().get(outId2);
  auto ops    = graph.getDirectViewChain(inT, outT);
  auto ops2   = graph.getDirectViewChain(inT, outT2);

  BOOST_CHECK(ops.first);
  BOOST_CHECK(ops.second[0]->opid == Onnx::CustomOperators::ReshapeInplace);
  BOOST_CHECK(ops.second[1]->opid == Onnx::CustomOperators::TransposeInplace);
  BOOST_CHECK(ops.second[2]->opid == Onnx::CustomOperators::IdentityInplace);

  BOOST_CHECK(!ops2.first && ops2.second.size() == 0);
}

// Similar to above, but with a call op. Check that the getDirectViewChain works
// within the subgraph

// clang-format off
//                              BuilderGraph_2
//          ┌───────────────────────────────────────────────────────────────┐
//          │                                                               │
//          │          ┌───► transpose───►reshape ─►transpose ──►identity───┼───► acts[0]
//          │          │                                                    │
//          │          │                                                    │
// SuperIn0 ├─► SuperIn0'                                                   │
//          │          │                                                    │
//          │          └───► reshape────►transpose──► cos───────────────────┼───► acts[1]
//          │                                                               │
//          └───────────────────────────────────────────────────────────────┘
// clang-format on

BOOST_AUTO_TEST_CASE(DirectViewChangeTest1) {

  auto superBuilder = Builder::create();
  auto builder      = &(superBuilder->createSubgraphBuilder());
  auto aiOnnx       = builder->aiOnnxOpset9();
  auto aiGraphcore  = builder->aiGraphcoreOpset1();

  TensorInfo shape0{"FLOAT", std::vector<int64_t>{2, 4}};

  auto superIn0 = superBuilder->addInputTensor(shape0, "superIn0");

  builder->addInputTensorFromParentGraph(superIn0);

  auto t0  = aiOnnx.transpose({superIn0});
  auto rs0 = aiGraphcore.reshape({t0}, {1, 8}, "rs0");
  auto t1  = aiOnnx.transpose({rs0}, {}, "ts1");
  t1       = aiOnnx.identity({t1}, "id1");

  auto rs1 = aiGraphcore.reshape({superIn0}, {1, 8}, "rs0");
  auto t2  = aiOnnx.transpose({rs0});
  auto c3  = aiOnnx.cos({t2}, "cos0");

  builder->addOutputTensor(t1);
  builder->addOutputTensor(c3);

  auto superAiGraphcore = superBuilder->aiGraphcoreOpset1();
  auto acts             = superAiGraphcore.call({superIn0}, 2, *builder);

  superBuilder->addOutputTensor(acts[0]);
  superBuilder->addOutputTensor(acts[1]);

  auto proto      = superBuilder->getModelProto();
  auto modelProto = io::getModelFromString(proto);

  // Create the IR
  auto dataFlow = DataFlow(1,
                           {{{acts[0], AnchorReturnType("ALL")},
                             {acts[1], AnchorReturnType("ALL")}}});
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

  auto &subG = ir.getGraph(GraphId("BuilderGraph_2"));

  auto superIn0T = ir.getMainGraph().getTensors().get(superIn0);
  auto subGinT   = subG.getTensors().get(addScope(subG, superIn0));
  auto actT0     = ir.getMainGraph().getTensors().get(acts[0]);
  auto actT1     = ir.getMainGraph().getTensors().get(acts[1]);

  auto cT  = subG.getTensors().get(addScope(subG, c3));
  auto t1T = subG.getTensors().get(addScope(subG, t1));

  auto ops = ir.getMainGraph().getDirectViewChain(superIn0T, actT0);
  // Call op is not viewChanging
  BOOST_CHECK(!ops.first && ops.second.size() == 0);

  auto ops2 = ir.getMainGraph().getDirectViewChain(superIn0T, actT1);
  // Call op is not viewChanging
  BOOST_CHECK(!ops.first && ops2.second.size() == 0);

  // Care to use the superIn0 from within the subgraph
  auto ops3 = subG.getDirectViewChain(subGinT, cT);
  // 'Blocked' by cos op
  BOOST_CHECK(!ops3.first && ops3.second.size() == 0);
  auto ops4 = subG.getDirectViewChain(subGinT, t1T);

  // This chain should work
  BOOST_CHECK(ops4.first);
  BOOST_CHECK(ops4.second[0]->opid == Onnx::CustomOperators::TransposeInplace);
  BOOST_CHECK(ops4.second[1]->opid == Onnx::CustomOperators::ReshapeInplace);
  BOOST_CHECK(ops4.second[2]->opid == Onnx::CustomOperators::TransposeInplace);
  BOOST_CHECK(ops4.second[3]->opid == Onnx::CustomOperators::IdentityInplace);

  BOOST_CHECK_THROW(subG.getDirectViewChain(superIn0T, actT0), popart::error);
  BOOST_CHECK_THROW(subG.getDirectViewChain(superIn0T, actT1), popart::error);
}
