// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE Add0InplaceTest

#include <../test_runner.hpp>
#include <boost/test/unit_test.hpp>
#include <elementwisebinary_testcase.hpp>
#include <popart/op/add.hpp>
#include <popart/patterns/pattern.hpp>
#include <popart/tensors.hpp>

using namespace popart;

// Test case class for testing inplacing of AddOp
class AddInplaceTestCase : public ElementWiseBinaryTestCase {
public:
  OperatorIdentifier basicOp() const final { return Onnx::AiOnnx::OpSet9::Add; }

  bool hasLhsOp() const final { return true; }

  OperatorIdentifier lhsOp() const final {
    return Onnx::CustomOperators::AddLhsInplace;
  }

  bool hasRhsOp() const final { return true; }

  OperatorIdentifier rhsOp() const final {
    return Onnx::CustomOperators::AddRhsInplace;
  }

  TensorId insertOp(AiOnnxOpset9 &opset,
                    const TensorId &a,
                    const TensorId &b) const final {
    return opset.add({a, b});
  }
};

// Basic case where arg0 and arg1 have matching shape
// (2x2) + (2x2)
BOOST_AUTO_TEST_CASE(Inplace_add0) {
  TestData data{/* A = */ {1, 2, 3, 4},
                /* B = */ {2, 3, 4, 5},
                /* out = */ {3, 5, 7, 9},
                /* shape(A) = */ {2, 2},
                /* shape(B) = */ {2, 2},
                /* shape(out) = */ {2, 2}};

  AddInplaceTestCase testCase;
  testCase.checkBasicOpIsInplaced(data);
}

// Arg0 is larger than arg1
// (2x2) + (1x2)
BOOST_AUTO_TEST_CASE(Inplace_add1) {
  TestData data{/* A = */ {1, 2, 3, 4},
                /* B = */ {2, 3},
                /* out = */ {3, 5, 5, 7},
                /* shape(A) = */ {2, 2},
                /* shape(B) = */ {1, 2},
                /* shape(out) = */ {2, 2}};

  AddInplaceTestCase testCase;
  testCase.checkBasicOpIsLhsInplaced(data);
}

// Arg1 is larger than arg0
// (1x2) + (2x2)
BOOST_AUTO_TEST_CASE(Inplace_add2) {
  TestData data{/* A = */ {2, 3},
                /* B = */ {1, 2, 3, 4},
                /* out = */ {3, 5, 5, 7},
                /* shape(A) = */ {1, 2},
                /* shape(B) = */ {2, 2},
                /* shape(out) = */ {2, 2}};

  AddInplaceTestCase testCase;
  testCase.checkBasicOpIsRhsInplaced(data);
}

// Arg0 and arg1 are of different ranks
// (2x2) + (2)
BOOST_AUTO_TEST_CASE(Inplace_add3) {
  TestData data{/* A = */ {1, 2, 3, 4},
                /* B = */ {2, 3},
                /* out = */ {3, 5, 5, 7},
                /* shape(A) = */ {2, 2},
                /* shape(B) = */ {2},
                /* shape(out) = */ {2, 2}};

  AddInplaceTestCase testCase;
  testCase.checkBasicOpIsLhsInplaced(data);
}

// Arg0 and arg1 are of different ranks
// (2) + (2x2)
BOOST_AUTO_TEST_CASE(Inplace_add4) {
  TestData data{/* A = */ {2, 3},
                /* B = */ {1, 2, 3, 4},
                /* out = */ {3, 5, 5, 7},
                /* shape(A) = */ {2},
                /* shape(B) = */ {2, 2},
                /* shape(out) = */ {2, 2}};

  AddInplaceTestCase testCase;
  testCase.checkBasicOpIsRhsInplaced(data);
}

// Checking AddOp fwdRegMap
BOOST_AUTO_TEST_CASE(Add_fwdRegMap0) { AddInplaceTestCase().checkFwdRegMap(); }

// Checking AddOp fwdRegMap
BOOST_AUTO_TEST_CASE(Add_bwdRegMap0) { AddInplaceTestCase().checkBwdRegMap(); }

// Check AddOp::inplacePriorityDefault priorities
// sides that connect two convolutions
BOOST_AUTO_TEST_CASE(Inplace_add5) {
  auto run_test = [&](OperatorIdentifier opid, bool include_relu) {
    TensorInfo info_1188{"FLOAT", std::vector<int64_t>{1, 1, 8, 8}};
    TensorInfo info_1144{"FLOAT", std::vector<int64_t>{1, 1, 4, 4}};
    TensorInfo info_1122{"FLOAT", std::vector<int64_t>{1, 1, 2, 2}};

    TestRunner runner;
    runner.patterns.enableInPlace(true);
    runner.patterns.enableUpdateInplacePrioritiesForIpu(true);

    runner.buildModel([&](Builder &builder) {
      auto aiOnnx = builder.aiOnnxOpset9();
      auto in0    = builder.addInputTensor(info_1188);
      auto w0     = builder.addInputTensor(info_1122);
      auto i1     = aiOnnx.identity({builder.addInputTensor(info_1144)});
      auto w1     = aiOnnx.identity({builder.addInputTensor(info_1122)});
      auto c0 = aiOnnx.conv({in0, w0}, {1, 1}, 1, {2, 2}, {0, 0, 0, 0}, {2, 2});

      // Order the inputs depending on whether we want the lhs inplace or rhs
      // inplace version
      auto a0 = [&]() {
        if (opid == Onnx::CustomOperators::AddLhsInplace) {
          return aiOnnx.add({c0, i1});
        } else {
          return aiOnnx.add({i1, c0});
        }
      }();

      if (include_relu) {
        a0 = aiOnnx.relu({a0});
      }

      auto c1 = aiOnnx.conv({a0, w1}, {1, 1}, 1, {2, 2}, {0, 0, 0, 0}, {2, 2});

      auto out = aiOnnx.identity({c1});
      builder.addOutputTensor(out);

      return out;
    });

    runner.checkIr([&](Ir &ir) {
      BOOST_CHECK(ir.opsOfType(Onnx::AiOnnx::OpSet9::Add).size() == 0);
      BOOST_CHECK(ir.opsOfType(opid).size() == 1);
    });
  };

  run_test(Onnx::CustomOperators::AddLhsInplace, false);
  run_test(Onnx::CustomOperators::AddRhsInplace, false);
  run_test(Onnx::CustomOperators::AddLhsInplace, true);
  run_test(Onnx::CustomOperators::AddRhsInplace, true);
}
