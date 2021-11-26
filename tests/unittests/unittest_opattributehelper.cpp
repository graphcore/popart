// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#define BOOST_TEST_MODULE BuilderTest

#include <boost/test/unit_test.hpp>

#include <popart/alias/aliasmodelgrower.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/names.hpp>
#include <popart/op/add.hpp>
#include <popart/op/call.hpp>
#include <popart/op/init.hpp>
#include <popart/op/ipucopy.hpp>
#include <popart/opattributehelper.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/tensornames.hpp>
#include <popart/tensors.hpp>

using namespace popart;

/**
 * Model describing the graph, creating multiple typical situations where
 * InheritOpAttributeHelper can be tested on:
 *
 * \code
 * (A) [initOp0]
 *  |    |
 *  |   (B)
 *  |    |
 * [addOp0]  (C)
 *    |       |
 *   (D)      |
 *    |       |
 * .-------------subgraph--.
 * | (E)     (F)           |
 * |  |      /             |
 * | [addOp1]  [initOp1]   |
 * |    |        |         |
 * |   (G)      (H)        |
 * '-----------------------'
 *      |       /
 *     (I)    (J)
 *      |     /  \
 *    [addOp2]   [ipuCopyOp0]
 *      |         |
 *     (K)       (L)
 *      |         |
 *  [ipuCopyOp1]  |
 *      |        /
 *     (M)      /
 *      |      /
 *     [addOp3]
 *      |
 *     (N)
 *      |
 *   {anchor}
 *
 * \endcode
 *
 */
struct TestModel {

  TestModel() : graph(ir.getMainGraph()), subgraph(ir.createGraph({"sg0"})) {
    // Set up the graphs
    // Manipulate settings
    SessionOptions options;

    // Ensures we can use execution phase and virtual graph attributes
    options.virtualGraphMode              = VirtualGraphMode::ExecutionPhases;
    options.executionPhaseSettings.phases = 4;

    // Ensures we can use pipeline attributes
    options.enablePipelining = true;

    ir.setUserOptions(options);

    Op::Settings gSettings(graph, "op", graph.getScope());
    Op::Settings sgSettings(subgraph, "op", subgraph.getScope());

    auto art = AnchorReturnType("All");
    TensorInfo tInfo{DataType::FLOAT, {4}};
    float tData[] = {0, 1, 2, 3};

    subgraph.addInput(addScope(subgraph, "E"), tInfo);
    subgraph.addInput(addScope(subgraph, "F"), tInfo);

    addOp1 = subgraph.createConnectedOp<AddOp>(
        {{AddOp::getArg0InIndex(), addScope(subgraph, "E")},
         {AddOp::getArg1InIndex(), addScope(subgraph, "F")}},
        {{AddOp::getOutIndex(), addScope(subgraph, "G")}},
        Onnx::Operators::Add_7,
        sgSettings);

    graph.getTensors().addVarInit("A", tInfo, static_cast<void *>(&tData));
    graph.getTensors().addVarInit("C", tInfo, static_cast<void *>(&tData));

    initOp0 = graph.createConnectedOp<InitOp>({},
                                              {{InitOp::getOutIndex(), "B"}},
                                              Onnx::CustomOperators::Init_1,
                                              tInfo,
                                              TensorType::ActGrad,
                                              InitType::Zero,
                                              gSettings);

    initOp1 = subgraph.createConnectedOp<InitOp>(
        {},
        {{InitOp::getOutIndex(), addScope(subgraph, "H")}},
        Onnx::CustomOperators::Init_1,
        tInfo,
        TensorType::ActGrad,
        InitType::Zero,
        sgSettings);

    subgraph.markAsOutput(addScope(subgraph, "G"));
    subgraph.markAsOutput(addScope(subgraph, "H"));

    addOp0 = graph.createConnectedOp<AddOp>(
        {{AddOp::getArg0InIndex(), "A"}, {AddOp::getArg1InIndex(), "B"}},
        {{AddOp::getOutIndex(), "D"}},
        Onnx::Operators::Add_7,
        gSettings);

    callOp = graph.createConnectedOp<CallOp>({{0, "D"}, {1, "C"}},
                                             {{0, "I"}, {1, "J"}},
                                             Onnx::CustomOperators::Call_1,
                                             subgraph,
                                             gSettings);

    addOp2 = graph.createConnectedOp<AddOp>(
        {{AddOp::getArg0InIndex(), "I"}, {AddOp::getArg1InIndex(), "J"}},
        {{AddOp::getOutIndex(), "K"}},
        Onnx::Operators::Add_7,
        gSettings);

    ipuCopyOp0 =
        graph.createOp<IpuCopyOp>(Onnx::CustomOperators::IpuCopy, 1, gSettings);
    ipuCopyOp0->connectInTensor(0, "J", 0);
    ipuCopyOp0->createAndConnectOutTensor(0, "L");
    ipuCopyOp0->setup();

    ipuCopyOp1 =
        graph.createOp<IpuCopyOp>(Onnx::CustomOperators::IpuCopy, 1, gSettings);
    ipuCopyOp1->connectInTensor(0, "K", 0);
    ipuCopyOp1->createAndConnectOutTensor(0, "M");
    ipuCopyOp1->setup();

    addOp3 = graph.createConnectedOp<AddOp>(
        {{AddOp::getArg0InIndex(), "M"}, {AddOp::getArg1InIndex(), "L"}},
        {{AddOp::getOutIndex(), "N"}},
        Onnx::Operators::Add_7,
        gSettings);

    auto df = DataFlow(1, {{"N", art}});
    ir.setDataFlow(df);

    AliasModelGrower aliasModelGrower{aliasModel};
    aliasModelGrower.growFullGraph(graph, DataDependenciesOnly::Yes);
    aliasModelGrower.growFullGraph(subgraph, DataDependenciesOnly::Yes);
  }

  popart::Ir ir;
  Graph &graph;
  Graph &subgraph;

  // Ops
  InitOp *initOp0 = nullptr;
  InitOp *initOp1 = nullptr;

  AddOp *addOp0 = nullptr;
  AddOp *addOp1 = nullptr;
  AddOp *addOp2 = nullptr;
  AddOp *addOp3 = nullptr;

  CallOp *callOp = nullptr;

  IpuCopyOp *ipuCopyOp0 = nullptr;
  IpuCopyOp *ipuCopyOp1 = nullptr;

  AliasModel aliasModel;
};

// Test inheriting attributes to addOp2
BOOST_AUTO_TEST_CASE(testOpAttributeHelperAddOp2) {

  TestModel model;

  model.callOp->settings.executionContext =
      ExecutionContext::AccumulateOuterFragment;
  model.callOp->settings.batchSerializedPhase = 1;
  model.callOp->settings.vgraphId             = 2;
  model.addOp1->settings.vgraphId             = 0;
  model.addOp1->settings.batchSerializedPhase = 2;
  model.initOp1->settings.vgraphId            = 1;

  // Let addOp2 inherit
  InheritOpAttributeHelper::apply(model.addOp2, true, model.aliasModel);

  // Expect the highest VGID on the producer side. The VGID 2 on callOp is
  // ignored due to virtual graph introspection resolving to addOp1 and
  // initOp1 inside the graph instead.
  BOOST_CHECK_EQUAL(model.addOp2->settings.vgraphId, OptionalVGraphId{1});

  // Highest batch serialized phase on the same graph level (from callOp)
  BOOST_CHECK_EQUAL(model.addOp2->settings.batchSerializedPhase,
                    OptionalBatchSerializedPhase{1});

  // Highest ExecutionContext on the same graph level (from CallOp)
  BOOST_CHECK_EQUAL(model.addOp2->settings.executionContext,
                    ExecutionContext::AccumulateOuterFragment);
}

// Test inheriting pipeline stage attribute from both consumer and producer side
BOOST_AUTO_TEST_CASE(testOpAttributeHelperPipeline) {

  TestModel model;

  model.ipuCopyOp0->settings.pipelineStage = 2;
  model.ipuCopyOp1->settings.pipelineStage = 3;

  // Let addOp2 and addOp3 inherit
  InheritOpAttributeHelper::apply(model.addOp2, true, model.aliasModel);
  InheritOpAttributeHelper::apply(model.addOp3, true, model.aliasModel);

  // Result is an inconsistency on VGIDs and PipelineStages due to ipuCopyOp1
  // once being a producer and once a consumer
  BOOST_CHECK_EQUAL(model.addOp2->settings.vgraphId, OptionalVGraphId{0});
  BOOST_CHECK_EQUAL(model.addOp2->settings.pipelineStage,
                    OptionalPipelineStage{3});

  BOOST_CHECK_EQUAL(model.addOp3->settings.vgraphId, OptionalVGraphId{1});
  BOOST_CHECK_EQUAL(model.addOp3->settings.pipelineStage,
                    OptionalPipelineStage{3});

  // Add more annotations on callOp for pipelineStage and VGID (within callOp).
  // Demonstrates also that setting the pipelineStage inside the callOp has no
  // effect. When inheriting from a producer, the highest pipelineStage is
  // inherited, but only on the same graph, not from subgraphs.
  model.callOp->settings.pipelineStage  = 2;
  model.addOp1->settings.vgraphId       = 0;
  model.addOp1->settings.pipelineStage  = 4;
  model.initOp1->settings.vgraphId      = 0;
  model.initOp1->settings.pipelineStage = 4;

  // Inherit again
  InheritOpAttributeHelper::apply(model.addOp2, true, model.aliasModel);

  // Check that the VGIDs are now consistent.
  BOOST_CHECK_EQUAL(model.addOp2->settings.vgraphId, OptionalVGraphId{0});
  BOOST_CHECK_EQUAL(model.addOp2->settings.pipelineStage,
                    OptionalPipelineStage{2});

  BOOST_CHECK_EQUAL(model.addOp3->settings.vgraphId, OptionalVGraphId{1});
  BOOST_CHECK_EQUAL(model.addOp3->settings.pipelineStage,
                    OptionalPipelineStage{3});
}

// Test inheriting VGID without actually setting it on any Op (via IpuCopyOp)
BOOST_AUTO_TEST_CASE(testOpAttributeHelperVGIDIpuCopyOp) {
  TestModel model;
  InheritOpAttributeHelper::apply(model.initOp0, true, model.aliasModel);
  InheritOpAttributeHelper::apply(model.initOp1, true, model.aliasModel);
  InheritOpAttributeHelper::apply(model.addOp0, true, model.aliasModel);
  InheritOpAttributeHelper::apply(model.addOp1, true, model.aliasModel);
  InheritOpAttributeHelper::apply(model.addOp2, true, model.aliasModel);
  InheritOpAttributeHelper::apply(model.addOp3, true, model.aliasModel);
  InheritOpAttributeHelper::apply(model.callOp, true, model.aliasModel);

  // The VGID will be defined dependent on if the Op is connected before or
  // after the IpuCopyOps
  BOOST_CHECK_EQUAL(model.initOp0->settings.vgraphId, OptionalVGraphId{0});
  BOOST_CHECK_EQUAL(model.initOp1->settings.vgraphId, OptionalVGraphId{0});
  BOOST_CHECK_EQUAL(model.addOp0->settings.vgraphId, OptionalVGraphId{0});
  BOOST_CHECK_EQUAL(model.addOp1->settings.vgraphId, OptionalVGraphId{0});
  BOOST_CHECK_EQUAL(model.addOp2->settings.vgraphId, OptionalVGraphId{0});
  BOOST_CHECK_EQUAL(model.addOp3->settings.vgraphId, OptionalVGraphId{1});
  BOOST_CHECK_EQUAL(model.callOp->settings.vgraphId, OptionalVGraphId{0});
}

// Test inheriting executionPhase, including adjustments due to IpuCopyOp
BOOST_AUTO_TEST_CASE(testOpAttributeHelperExecutionPhase) {
  TestModel model;

  model.ipuCopyOp0->settings.executionPhase = 1;
  model.ipuCopyOp1->settings.executionPhase = 3;

  InheritOpAttributeHelper::apply(model.addOp2, true, model.aliasModel);
  InheritOpAttributeHelper::apply(model.addOp3, true, model.aliasModel);

  // Lowest execution phase annotated after addOp2
  // (ipuCopy0 is a sibling to addOp2 in the graph, and so has lower priority
  // than a preceding producer or following consumer when inheriting attributes)
  BOOST_CHECK_EQUAL(model.addOp2->settings.executionPhase,
                    OptionalExecutionPhase{3});

  // The maximum phase encountered before is 3, but since the provider of that
  // attribute is an inter-phase copy, wit addOp3 as a downstream consumer,
  // the phase is shifted to 4.
  BOOST_CHECK_EQUAL(model.addOp3->settings.executionPhase,
                    OptionalExecutionPhase{4});
}

// Test inheriting from downstream consumers only
BOOST_AUTO_TEST_CASE(testOpAttributeHelperInheritConsumerSubgraph) {
  TestModel model;

  // Inside the subgraph
  model.addOp1->settings.vgraphId          = 2;
  model.addOp1->settings.executionContext  = ExecutionContext::Subgraph;
  model.initOp1->settings.vgraphId         = 2;
  model.initOp1->settings.executionContext = ExecutionContext::Subgraph;

  // The CallOp itself, plus the preceding InitOp
  model.callOp->settings.executionContext =
      ExecutionContext::WeightsToHostFragment;
  model.initOp0->settings.executionContext =
      ExecutionContext::WeightsFromHostFragment;

  InheritOpAttributeHelper::apply(model.addOp0, true, model.aliasModel);

  BOOST_CHECK_EQUAL(model.addOp0->settings.vgraphId, OptionalVGraphId{2});
  BOOST_CHECK_EQUAL(model.addOp0->settings.executionContext,
                    ExecutionContext::WeightsFromHostFragment);
}
