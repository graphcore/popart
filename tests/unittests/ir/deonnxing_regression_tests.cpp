// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE Test_Ir_DeonnxingRegressionTests
#include <boost/test/unit_test.hpp>

#include <popart/error.hpp>
#include <popart/graph.hpp>
#include <popart/inputshapeinfo.hpp>
#include <popart/ir.hpp>
#include <popart/tensorinfo.hpp>

#include <onnx/onnx_pb.h>

#include <string>

using namespace popart;

namespace {

bool hasPrefix(const std::string &str, const std::string &prefix) {
  return str.length() >= prefix.length() &&
         str.compare(0, prefix.size(), prefix) == 0;
}

template <typename Ex>
std::function<bool(const Ex &)>
checkErrorMsgHasPrefixFn(const std::string &prefix) {
  return [=](const Ex &ex) -> bool { return hasPrefix(ex.what(), prefix); };
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
