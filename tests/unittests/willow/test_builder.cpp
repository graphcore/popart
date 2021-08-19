// Copyright (c) 2019 Graphcore Ltd. All rights reserved.

#define BOOST_TEST_MODULE BuilderTest

#include <boost/test/unit_test.hpp>

#include <filereader.hpp>
#include <popart/builder.hpp>
#include <popart/error.hpp>
#include <popart/inputshapeinfo.hpp>
#include <popart/names.hpp>
#include <popart/op/l1.hpp>
#include <popart/op/nll.hpp>
#include <popart/opmanager.hpp>
#include <popart/popx/opx.hpp>
#include <popart/shapeinference.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/tensornames.hpp>
#include <popart/tensors.hpp>

using namespace popart;

bool invalidOpsetId(const error &ex) {
  BOOST_CHECK_EQUAL(
      ex.what(),
      std::string("Invalid opset 6 used to add an operation. Opset for domain "
                  "ai.onnx already defined as 9"));
  return true;
}

BOOST_AUTO_TEST_CASE(Builder_MultiOpset) {

  // Build a model using two opset for the same domain, this will be invalid

  // Build an onnx model
  auto builder = Builder::create();
  auto aiOnnx9 = builder->aiOnnxOpset9();
  auto aiOnnx6 = builder->aiOnnxOpset6();

  TensorInfo shape0{"FLOAT", std::vector<int64_t>{3}};
  auto in0 = builder->addInputTensor(shape0);
  auto in1 = builder->addInputTensor(shape0);
  auto t1  = aiOnnx9.concat({in0, in1}, 0);

  auto in2 = builder->addInputTensor(shape0);

  BOOST_CHECK_EXCEPTION(aiOnnx6.concat({in2, t1}, 0), error, invalidOpsetId);
}

namespace CustomOperators {
const OperatorIdentifier Foo = {"com.acme", "Foo", 1};
} // namespace CustomOperators

// An IdentityOp
class FooOp : public Op {
public:
  FooOp(const OperatorIdentifier &_opid, const Op::Settings &settings_)
      : Op(_opid, settings_) {}

  void setup() final { outInfo(0) = inInfo(0); }

  std::unique_ptr<Op> clone() const final {
    return std::make_unique<FooOp>(*this);
  }
  float getSubgraphValue() const final { return getLowSubgraphValue(); }
};

static popart::OpDefinition fooOpDef(
    {popart::OpDefinition::Inputs({
         {"input", {{popart::DataType::FLOAT, popart::DataType::FLOAT16}}},
     }),
     popart::OpDefinition::Outputs(
         {{"output", {{popart::DataType::FLOAT, popart::DataType::FLOAT16}}}}),
     popart::OpDefinition::Attributes({})});

static OpCreator<FooOp> fooOpCreator({{CustomOperators::Foo, fooOpDef}});

class FooOpx : public popx::Opx {
public:
  FooOpx(Op *op, popx::Devicex *devicex) : popx::Opx(op, devicex) {
    verifyOp<FooOp>(op, CustomOperators::Foo);
  }

  void grow(poplar::program::Sequence &prog) const final {
    insert(outId(0), cloneNcopy(prog, getInTensor(0)));
  }
};

bool noTensorShape(const error &ex) {
  // Error thrown inside BuilderImpl::getValueInfoProto
  std::string strerr = " is not an known tensor. Must be one of ";
  return std::string(ex.what()).find(strerr) != std::string::npos;
}

BOOST_AUTO_TEST_CASE(Builder_CustomOp_Into_WindowParameter_Op) {
  // Build the model:
  //
  // in0 ----> Foo ----> t0 ---> MaxPool --> o
  //       (Custom Op)
  //
  // in order to test that (until T17932 is complete) a tensor generated
  // by a custom op (which lacks an onnx shape inference method) can be
  // an input to an op whose Builder method calls into verify_AiOnnxOpset...

  // Build an onnx model
  auto builder = Builder::create();
  auto aiOnnx  = builder->aiOnnxOpset6();

  auto in0 = builder->addInputTensor("FLOAT", {1, 1, 10, 10});
  auto t0  = builder->customOp(CustomOperators::Foo, 1, {in0}, 1, {}).at(0);

  // Confirm that the output of the custom op does not have a known tensor shape
  BOOST_CHECK_EXCEPTION(builder->getTensorShape(t0), error, noTensorShape);

  // Confirm that maxpool (an op whose builder method calls into
  // verifyWindowParameters) can take an input without a known tensor shape
  auto o = aiOnnx.maxpool({t0}, {3, 3}, {}, {});
}

// Testing shape inference functions.
namespace CustomOperators {
const popart::OperatorIdentifier Bar = {"com.acme", "Bar", 1};
} // namespace CustomOperators

static popart::RegisterShapeInferenceFunction
    barOpShapeInference(CustomOperators::Bar, [](ShapeInferenceContext &ctx) {
      int64_t k = ctx.getAttribute<Attributes::Int>("k");

      auto shape = ctx.inShape(0);
      for (int i = 0; i < shape.size(); i++) {
        shape.at(i) *= k;
      }
      ctx.outInfo(0) = {ctx.inType(0), shape};
    });

void check_equal(const std::vector<int64_t> &v0,
                 const std::vector<int64_t> &v1) {
  BOOST_CHECK_EQUAL(v0.size(), v1.size());
  for (int i = 0; i < v0.size(); i++) {
    BOOST_CHECK_EQUAL(v0.at(i), v1.at(i));
  }
}

BOOST_AUTO_TEST_CASE(Builder_CustomOp_ShapeInference) {
  // Build the model:
  //
  // in0 ----> Bar ----> Reshape --> o
  //       (Custom Op)
  //
  // First check custom shape inference works for Bar, and then make sure onnx
  // shape inference works for an op after Bar.

  // Build an onnx model
  auto builder = Builder::create();
  auto aiOnnx  = builder->aiOnnxOpset9();
  auto in0     = builder->addInputTensor("FLOAT", {1, 2});

  // Add Bar op and check shape and type.
  auto t0 =
      builder->customOp(CustomOperators::Bar, 1, {in0}, 1, {{"k", int64_t{3}}})
          .at(0);
  check_equal(builder->getTensorShape(t0), {3, 6});
  BOOST_CHECK_EQUAL(builder->getTensorDataType(t0), DataType::FLOAT);

  // Add Reshape op and check shape and type.
  Shape outShape             = {3 * 6};
  Shape outShapeSize         = {static_cast<int64_t>(outShape.size())};
  ConstVoidData outShapeData = {outShape.data(), {"INT64", outShapeSize}};
  auto newShapeId            = aiOnnx.constant(outShapeData, "outShapeData");
  auto t1                    = aiOnnx.reshape({t0, newShapeId});
  check_equal(builder->getTensorShape(t1), {3 * 6});
  BOOST_CHECK_EQUAL(builder->getTensorDataType(t1), DataType::FLOAT);
}

void check_operator(const TensorId &output, Builder &builder) {
  float availableMemoryProportion = 0.75;
  builder.setAvailableMemoryProportion(output, availableMemoryProportion);
  float actualMemoryProp =
      builder.getFloatNodeAttribute(sAvailMemAttribute, {output});
  BOOST_CHECK_EQUAL(actualMemoryProp, availableMemoryProportion);
}

bool invalidProportion(const error &ex) {
  BOOST_CHECK_EQUAL(ex.what(), "availableMemoryProportion must be in (0,1]");
  return true;
}

BOOST_AUTO_TEST_CASE(Builder_SetAvailableMemoryProportion) {
  auto builder     = Builder::create();
  auto aiOnnx      = builder->aiOnnxOpset11();
  auto aiGraphcore = builder->aiGraphcoreOpset1();

  TensorInfo shape0{"FLOAT", std::vector<int64_t>{3, 3, 1, 1}};
  auto in0 = builder->addInputTensor(shape0);
  auto in1 = builder->addInputTensor(shape0);

  // Check the operators that support available memory proportion
  check_operator(aiOnnx.gather({in0, in1}), *builder);
  check_operator(aiOnnx.matmul({in0, in1}), *builder);
  check_operator(aiOnnx.conv({in0, in1}), *builder);
  check_operator(aiGraphcore.scatterreduce({in0, in1}, 1), *builder);

  // Check an op that doesn't support available memory proportion is a no-op
  auto op = aiOnnx.identity({in0});
  builder->setAvailableMemoryProportion(op, 0.2);
  BOOST_CHECK_MESSAGE(
      !builder->nodeHasAttribute(sAvailMemAttribute, {op}),
      "Identity operator should not have the available memory attribute");

  // Check out-of-range values for available memory proportion
  op = aiOnnx.gather({in0, in1});
  BOOST_CHECK_EXCEPTION(
      builder->setAvailableMemoryProportion(op, 1.1), error, invalidProportion);
  BOOST_CHECK_EXCEPTION(
      builder->setAvailableMemoryProportion(op, 0), error, invalidProportion);
}
