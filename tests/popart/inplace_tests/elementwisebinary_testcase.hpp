// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_ELEMENTWISEBINARY_TESTCASE_HPP
#define GUARD_ELEMENTWISEBINARY_TESTCASE_HPP

#include <../test_runner.hpp>
#include <boost/test/unit_test.hpp>

#include <popart/builder.hpp>
#include <popart/error.hpp>
#include <popart/ir.hpp>
#include <popart/names.hpp>
#include <popart/op/elementwise.hpp>
#include <popart/opidentifier.hpp>
#include <popart/tensorinfo.hpp>

#include <vector>

// Struct containing data inputs and expected output for testing element-wise
// binary operations
struct TestData {
  std::vector<float> InputA;
  std::vector<float> InputB;
  std::vector<float> Output;

  std::vector<int64_t> ShapeA;
  std::vector<int64_t> ShapeB;
  std::vector<int64_t> ShapeOutput;
};

// Abstract test case for checking inplacing of element-wise binary operations.
class ElementWiseBinaryTestCase {
public:
  // OperatorIdentifier for the "basic" operation -> not evaluated in place.
  virtual OperatorIdentifier basicOp() const = 0;

  // Does the op being tested have an LHS inplace variant?
  virtual bool hasLhsOp() const { return false; }

  // OperatorIdentifier for the LHS in place variant
  virtual OperatorIdentifier lhsOp() const {
    throw internal_error(
        "Operator under test does not have an LHS inplace variant");
  };

  // Does the op being tested have an RHS inplace variant?
  virtual bool hasRhsOp() const { return false; }

  // OperatorIdentifier for the RHS in place variant
  virtual OperatorIdentifier rhsOp() const {
    throw internal_error(
        "Operator under test does not have an RHS inplace variant");
  };

  // Inserts the op under test given two input tensors.
  virtual TensorId
  insertOp(AiOnnxOpset9 &, const TensorId &, const TensorId &) const = 0;

  virtual ~ElementWiseBinaryTestCase() {}

  // Check if the basic op has been replaced by the LHS inplace variant
  bool isLhsInplaced(Ir &ir) const {
    return numBasicOps(ir) == 0 && numLhsOps(ir) == 1 && numRhsOps(ir) == 0;
  }

  // Check if the basic op has been replaced by the RHS inplace variant
  bool isRhsInplaced(Ir &ir) const {
    return numBasicOps(ir) == 0 && numLhsOps(ir) == 0 && numRhsOps(ir) == 1;
  }

  // Check if the basic op has been replaced by either the LHS or RHS inplace
  // variants
  bool isInplaced(Ir &ir) const {
    return (hasLhsOp() && isLhsInplaced(ir)) ||
           (hasRhsOp() && isRhsInplaced(ir));
  }

  // Test method that verifies that the basic op has been replaced by either the
  // LHS or RHS inplace variants
  void checkBasicOpIsInplaced(const TestData &data) const {
    checkBasicOpInplacing(data, &ElementWiseBinaryTestCase::isInplaced);
  }

  // Test method that verifies that the basic op has been replaced by the LHS
  // inplace variant
  void checkBasicOpIsLhsInplaced(const TestData &data) const {
    BOOST_ASSERT_MSG(
        hasLhsOp(), "Operator under test does not have an LHS inplace variant");
    checkBasicOpInplacing(data, &ElementWiseBinaryTestCase::isLhsInplaced);
  }

  // Test method that verifies that the basic op has been replaced by the RHS
  // inplace variant
  void checkBasicOpIsRhsInplaced(const TestData &data) const {
    BOOST_ASSERT_MSG(
        hasRhsOp(), "Operator under test does not have an RHS inplace variant");
    checkBasicOpInplacing(data, &ElementWiseBinaryTestCase::isRhsInplaced);
  }

  // Test method for fwdRegMap implementation
  void checkFwdRegMap() const {
    TensorInfo info0{"FLOAT", std::vector<int64_t>{4, 2, 4}};
    TensorInfo info1{"FLOAT", std::vector<int64_t>{1, 4}};
    std::string opOut;

    auto buildModel = [&](Builder &builder) {
      auto aiOnnx = builder.aiOnnxOpset9();
      auto in0    = builder.addInputTensor(info0);
      auto in1    = builder.addInputTensor(info1);
      opOut       = insertOp(aiOnnx, in0, in1);
      builder.addOutputTensor(opOut);

      return opOut;
    };

    auto checkIr = [&](Ir &ir) {
      auto tensor = ir.getTensors().get(opOut);
      auto op     = tensor->getProducer();

      auto arg0Map = op->fwdRegMap(ElementWiseBinaryBaseOp::getArg0InIndex(),
                                   ElementWiseBinaryBaseOp::getOutIndex());
      auto arg1Map = op->fwdRegMap(ElementWiseBinaryBaseOp::getArg1InIndex(),
                                   ElementWiseBinaryBaseOp::getOutIndex());

      // dim 0 and 1 broadcast
      auto r                = arg1Map({{0, 0}, {1, 2}});
      view::Region expected = {{0, 0, 0}, {4, 2, 2}};
      BOOST_CHECK(r.front() == expected);
    };

    TestRunner runner;
    runner.patterns.enableInPlace(false);
    runner.buildModel(buildModel);
    runner.checkIr(checkIr);
  }

  // Test method for fwdRegMap implementation
  void checkBwdRegMap() const {
    TensorInfo info0{"FLOAT", std::vector<int64_t>{5, 7}};
    TensorInfo info1{"FLOAT", std::vector<int64_t>{5, 1}};
    std::string opOut;

    auto buildModel = [&](Builder &builder) {
      auto aiOnnx = builder.aiOnnxOpset9();
      auto in0    = builder.addInputTensor(info0);
      auto in1    = builder.addInputTensor(info1);
      opOut       = insertOp(aiOnnx, in0, in1);
      builder.addOutputTensor(opOut);

      return opOut;
    };

    auto checkIr = [&](Ir &ir) {
      auto tensor = ir.getTensors().get(opOut);
      auto addOp  = tensor->getProducer();

      auto arg1Map = addOp->bwdRegMap(ElementWiseBinaryBaseOp::getArg1InIndex(),
                                      ElementWiseBinaryBaseOp::getOutIndex());

      // dim 0 and 1 broadcast
      auto r                = arg1Map({{1, 4}, {3, 6}});
      view::Region expected = {{1, 0}, {3, 1}};
      BOOST_CHECK(r.front() == expected);
    };

    TestRunner runner;
    runner.patterns.enableInPlace(false);
    runner.buildModel(buildModel);
    runner.checkIr(checkIr);
  }

private:
  size_t numBasicOps(Ir &ir) const { return ir.opsOfType(basicOp()).size(); }

  size_t numLhsOps(Ir &ir) const {
    return hasLhsOp() ? ir.opsOfType(lhsOp()).size() : 0;
  }

  size_t numRhsOps(Ir &ir) const {
    return hasRhsOp() ? ir.opsOfType(rhsOp()).size() : 0;
  }

  template <typename IrChecker>
  void checkBasicOpInplacing(const TestData &data,
                             IrChecker &&irChecker) const {
    TensorInfo infoA{"FLOAT", data.ShapeA};
    TensorInfo infoB{"FLOAT", data.ShapeB};
    TensorInfo infoOut{"FLOAT", data.ShapeOutput};

    std::vector<TestTensor> inputs;
    std::vector<TestTensor> outputs;

    auto buildModel = [&](Builder &builder) {
      auto aiOnnx = builder.aiOnnxOpset9();
      auto in0    = builder.addInputTensor(infoA);
      auto in1    = builder.addInputTensor(infoB);
      auto out    = insertOp(aiOnnx, in0, in1);
      builder.addOutputTensor(out);

      inputs.push_back(
          TestTensor::create<float>(in0, data.InputA, infoA.shape()));
      inputs.push_back(
          TestTensor::create<float>(in1, data.InputB, infoB.shape()));
      outputs.push_back(TestTensor::create<float>(out, infoOut.shape()));

      return out;
    };

    auto checkIr = [&](Ir &ir) {
      BOOST_CHECK_MESSAGE(
          (this->*irChecker)(ir),
          "IR Check Failed: did not contain the expected inplace ops");
    };

    auto checkResult = [&data](TestTensor &result) {
      auto actual = result.getDataCopy<float>();
      BOOST_CHECK_EQUAL_COLLECTIONS(
          actual.begin(), actual.end(), data.Output.begin(), data.Output.end());
    };

    TestRunner runner;
    runner.patterns.enableInPlace(true);
    runner.buildModel(buildModel);
    runner.checkIr(checkIr);
    runner.checkResult(checkResult, inputs, outputs);
  }
};

#endif
