// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE Test_Ir_DeonnxingRegressionTests
#include <algorithm>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/test/unit_test.hpp>
#include <functional>
#include <map>
#include <memory>
#include <onnx/onnx_pb.h>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include <popart/error.hpp>
#include <popart/graph.hpp>
#include <popart/inputshapeinfo.hpp>
#include <popart/ir.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/util.hpp>

#include "popart/builder.gen.hpp"
#include "popart/datatype.hpp"
#include "popart/erroruid.hpp"
#include "popart/graphid.hpp"
#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/operatoridentifier.hpp"
#include "popart/operators.hpp"
#include "popart/tensor.hpp"
#include "popart/tensordebuginfo.hpp"
#include "popart/tensornames.hpp"
#include "popart/tensors.hpp"

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

BOOST_AUTO_TEST_CASE(TestContainsInitialisersReturnsFalse) {
  Ir ir;
  bool res = true;
  BOOST_REQUIRE_NO_THROW((res = ir.containsInitialisers()));
  BOOST_REQUIRE(!res);
}

BOOST_AUTO_TEST_CASE(TestConfirmNoReservedIdsWorksWithoutOnnxModel) {
  const TensorInfo tInfo{DataType::FLOAT, {2}};
  Ir ir;

  InputShapeInfo isi;
  isi.add("unreserved-id-sdfsdfsdf", tInfo);
  ir.setInputShapeInfo(isi);

  BOOST_REQUIRE_NO_THROW(ir.confirmNoReservedIds());

  isi.add(reservedGradientPrefix() + std::string("some-id"), tInfo);
  ir.setInputShapeInfo(isi);

  // Error message is defined in Ir::confirmNonReservedId.
  const auto checkErrorFn = checkErrorMsgHasPrefixFn<error>(
      std::string("Provided tensor ") + reservedGradientPrefix());

  BOOST_REQUIRE_EXCEPTION(ir.confirmNoReservedIds(), error, checkErrorFn);
}

BOOST_AUTO_TEST_CASE(TestSetExternalTensorDataInfoThrows) {
  Ir ir;
  BOOST_REQUIRE_THROW(ir.setExternalTensorDataInfo("some tensor id", {}),
                      error);
}

BOOST_AUTO_TEST_CASE(TestTensorExistsInInitialiserReturnsFalseOnNoOnnxModel) {
  Ir ir;

  constexpr int nelms = 2;
  const TensorInfo ti(DataType::FLOAT, {nelms});
  std::vector<float> someData(nelms, 0.1f);

  const TensorId t = "t";

  ir.getMainGraph().getTensors().addVarInit(t, ti, someData.data());

  BOOST_REQUIRE(ir.tensorExistsInInitialisers(t) == false);
}

BOOST_AUTO_TEST_CASE(TestAddAdditionalModelProtoTensorsHandlesNoOnnxModel) {
  Ir ir;
  ir.addAdditionalModelProtoTensors();
}

BOOST_AUTO_TEST_CASE(TestAddAdditionalModelProtoTensorWorksButSavingThemFails) {
  // Calling addAdditionalModelProtoTensor to register the extra tensor should
  // work exactly the same, but then actually trying to save them with
  // addAdditionalModelProtoTensors should throw.

  Ir ir;

  constexpr int nelms = 2;
  const TensorInfo ti(DataType::FLOAT, {nelms});
  std::vector<float> someData(nelms, 0.1f);

  const TensorId t1 = "t1";
  const TensorId t2 = "t2";

  ir.getMainGraph().getTensors().addVarInit(t1, ti, someData.data());
  ir.getMainGraph().getTensors().addVarInit(t2, ti, someData.data());

  // Registering tensors does not throw.
  BOOST_REQUIRE_NO_THROW(ir.addAdditionalModelProtoTensor(t1));
  BOOST_REQUIRE_NO_THROW(ir.addAdditionalModelProtoTensor(ir.getTensor(t2)));

  // The tensors should be registered correctly.

  const auto &ts = ir.getAdditionalModelProtoTensors();

  BOOST_REQUIRE_EQUAL(ts.size(), 2);
  bool foundT1 = false;
  bool foundT2 = false;
  for (const auto t : ts) {
    if (t->id == t1) {
      foundT1 = true;
    } else if (t->id == t2) {
      foundT2 = true;
    }
  }
  BOOST_REQUIRE(foundT1);
  BOOST_REQUIRE(foundT2);

  // Trying to save the tensors does throw.
  BOOST_REQUIRE_THROW(ir.addAdditionalModelProtoTensors(), error);
}

BOOST_AUTO_TEST_CASE(TestRegisterInputTensors) {
  // Test does not throw if model set (even an empty one).
  {
    Ir ir;
    ir.setOnnxModel({});
    BOOST_REQUIRE_NO_THROW(ir.registerInputTensors());
  }

  // Test throws if model not set.
  {
    Ir ir;

    // Error message is defined in Ir::registerInputTensors.
    const auto checkErrorFn = checkErrorMsgHasPrefixFn<error>(
        "Ir::registerInputTensors: Ir has no Onnx model");

    BOOST_REQUIRE_EXCEPTION(ir.registerInputTensors(), error, checkErrorFn);
  }
}

BOOST_AUTO_TEST_CASE(TestConstructForwards) {
  // Test does not throw if model set (even an empty one).
  {
    Ir ir;
    ir.setOnnxModel({});
    BOOST_REQUIRE_NO_THROW(ir.constructForwards());
  }

  // Test throws if model not set.
  {
    Ir ir;

    // Error message is defined in Ir::constructForwards.
    const auto checkErrorFn = checkErrorMsgHasPrefixFn<error>(
        "Ir::constructForwards: Ir has no Onnx model");

    BOOST_REQUIRE_EXCEPTION(ir.constructForwards(), error, checkErrorFn);
  }
}

BOOST_AUTO_TEST_CASE(TestGetModel) {
  // Test does not throw if model set (even an empty one).
  {
    Ir ir;
    ir.setOnnxModel({});
    BOOST_REQUIRE_NO_THROW(ir.getModel());
  }

  // Test throws if model not set.
  {
    Ir ir;

    // Error message is defined in Ir::getModel.
    const auto checkErrorFn =
        checkErrorMsgHasPrefixFn<error>("Ir::getModel: Ir has no Onnx model");

    BOOST_REQUIRE_EXCEPTION(ir.getModel(), error, checkErrorFn);
  }
}

BOOST_AUTO_TEST_CASE(TestGetModelInputIdsReturnsEmpty) {
  Ir ir;

  std::vector<TensorId> ids;
  BOOST_REQUIRE_NO_THROW((ids = ir.getModelInputIds()));
  BOOST_REQUIRE(ids.empty());
}

BOOST_AUTO_TEST_CASE(TestGetOpSetVersionFromModelReturnsDefaultOpsetVersion) {
  Ir ir;
  BOOST_REQUIRE_EQUAL(ir.getOpSetVersionFromModel(Domain::ai_onnx),
                      ir.getDefaultOpsetVersion(Domain::ai_onnx));
  BOOST_REQUIRE_EQUAL(ir.getOpSetVersionFromModel(Domain::ai_onnx_ml),
                      ir.getDefaultOpsetVersion(Domain::ai_onnx_ml));
  BOOST_REQUIRE_EQUAL(ir.getOpSetVersionFromModel(Domain::ai_graphcore),
                      ir.getDefaultOpsetVersion(Domain::ai_graphcore));
}

#include <onnxutil.hpp>
#include <sstream>
#include <popart/builder.hpp>
#include <popart/op/call.hpp>
#include <popart/op/identity.hpp>

BOOST_AUTO_TEST_CASE(TestSerialise) {
  /*
    For the model:
      t -> CallOp -> u
             |
             \
              subgraph: [t -> Identity -> u]

    Assert:
      - If the main graph in the ONNX model has a custom name, the main graph's
        key in the json is that name, otherwise it is "maingraph".
      - Any other graph's key is graph->id.
   */

  const auto test = [](const auto initIrFn, const std::string &subgraphName) {
    Ir ir;
    initIrFn(ir);

    std::stringstream ss;
    ir.serialise(Ir::SerialiseFormat::JSON, ss);
    const auto modelStr = ss.str();

    // (rudimentarily) Check that there is one graph with the appropriate name
    // for the main graph, and one other graph with name `subgraphName`.

    // Note assuming the prefix "BuilderGraph_" is some pretty tight coupling,
    // but this is what the Ir code itself already did and we want to assert
    // this behaviour remains in place whilst refactoring the Ir.
    //
    // To future maintainers: If you want to change this behaviour of the Ir and
    // this test is failing, don't let it stop you - just change or delete this
    // test.

    const bool mainGraphHasCustomNameFromBuilder =
        ir.hasOnnxModel() && (ir.getModel().graph().name().find(
                                  "BuilderGraph_") == std::string::npos);

    const auto maingraphName = mainGraphHasCustomNameFromBuilder
                                   ? ir.getModel().graph().name()
                                   : "maingraph";

    BOOST_REQUIRE(modelStr.find("\"" + maingraphName + "\" :[") !=
                  std::string::npos);
    BOOST_REQUIRE(modelStr.find("\"" + subgraphName + "\" :[") !=
                  std::string::npos);
  };

  const auto irBuilderSubgraphId = "subgraph";
  const auto initIrBuilder       = [irBuilderSubgraphId](Ir &ir) {
    // 1. Build the Onnx model
    auto builder = Builder::create();

    TensorInfo tInfo(DataType::FLOAT, {2});
    const auto t = builder->addInputTensor(tInfo);

    auto &subBuilder = builder->createSubgraphBuilder();
    subBuilder.setGraphName(irBuilderSubgraphId);

    // Note t is inherited from parent scope.
    const auto subU = subBuilder.aiOnnxOpset7().identity({t});
    subBuilder.addOutputTensor(subU);

    constexpr int numSubgraphOutputs = 1;
    const auto outs =
        builder->aiGraphcoreOpset1().call({}, numSubgraphOutputs, subBuilder);
    const auto u = std::move(outs[0]);

    builder->addOutputTensor(u);

    // 2. Initialise Ir from Onnx model
    ir.setOnnxModel(onnxutil::getModelProto(builder->getModelProto()));
    ir.registerInputTensors();
    ir.constructForwards();
  };
  test(initIrBuilder, irBuilderSubgraphId);

  // Same as above except we set a custom maingraph name in the builder.
  const auto initIrBuilderWithCustomMainGraphName = [irBuilderSubgraphId](
                                                        Ir &ir) {
    // 1. Build the Onnx model
    auto builder = Builder::create();

    // ===== Set a custom name in the ONNX model =====
    builder->setGraphName("some_crazy_name");

    TensorInfo tInfo(DataType::FLOAT, {2});
    const auto t = builder->addInputTensor(tInfo);

    auto &subBuilder = builder->createSubgraphBuilder();
    subBuilder.setGraphName(irBuilderSubgraphId);

    // Note t is inherited from parent scope.
    const auto subU = subBuilder.aiOnnxOpset7().identity({t});
    subBuilder.addOutputTensor(subU);

    constexpr int numSubgraphOutputs = 1;
    const auto outs =
        builder->aiGraphcoreOpset1().call({}, numSubgraphOutputs, subBuilder);
    const auto u = std::move(outs[0]);

    builder->addOutputTensor(u);

    // 2. Initialise Ir from Onnx model
    ir.setOnnxModel(onnxutil::getModelProto(builder->getModelProto()));
    ir.registerInputTensors();
    ir.constructForwards();
  };
  test(initIrBuilderWithCustomMainGraphName, irBuilderSubgraphId);

  const auto irDirectSubgraphId = "another_subgraph_id";
  const auto initIrDirect       = [irDirectSubgraphId](Ir &ir) {
    auto &mainGraph = ir.getMainGraph();

    TensorInfo tInfo(DataType::FLOAT, {2});
    const auto t = mainGraph.addInput(tInfo);
    const auto u = "u";

    auto &subGraph  = ir.createGraph(GraphId{irDirectSubgraphId});
    const auto subT = subGraph.addInput(tInfo);
    const auto subU = addScope(subGraph, "u");

    subGraph.createConnectedOp<IdentityOp>({{IdentityOp::getInIndex(), subT}},
                                           {{IdentityOp::getOutIndex(), subU}},
                                           Onnx::Operators::Identity_1,
                                           Op::Settings{subGraph, "id"});
    subGraph.markAsOutput(subU);

    mainGraph.createConnectedOp<CallOp>({{0, t}},
                                        {{0, u}},
                                        Onnx::CustomOperators::Call_1,
                                        subGraph,
                                        Op::Settings{mainGraph, "call"});
    mainGraph.markAsOutput(u);
  };
  test(initIrDirect, irDirectSubgraphId);
}
