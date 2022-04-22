// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE Mul0InplaceTest

#include <algorithm>
#include <boost/test/unit_test.hpp>
#include <cstdint>
#include <string>

#include "elementwisebinary_testcase.hpp"
#include "popart/builder.gen.hpp"
#include "popart/names.hpp"
#include "popart/operatoridentifier.hpp"
#include "popart/operators.hpp"
#include "popart/tensordebuginfo.hpp"

using namespace popart;

// Test case class for testing inplacing of MulOp
class MulInplaceTestCase : public ElementWiseBinaryTestCase {
public:
  OperatorIdentifier basicOp() const final { return Onnx::AiOnnx::OpSet9::Mul; }

  bool hasLhsOp() const final { return true; }

  OperatorIdentifier lhsOp() const final {
    return Onnx::CustomOperators::MulLhsInplace;
  }

  bool hasRhsOp() const final { return true; }

  OperatorIdentifier rhsOp() const final {
    return Onnx::CustomOperators::MulRhsInplace;
  }

  TensorId insertOp(AiOnnxOpset9 &opset,
                    const TensorId &a,
                    const TensorId &b) const final {
    return opset.mul({a, b});
  }
};

// Basic case where arg0 and arg1 have matching shape
// (2x2) * (2x2)
// Expect that the IR does not contain the basic op and was replaced by either
// the LHS or RHS inplace variant.
BOOST_AUTO_TEST_CASE(Inplace_mul0) {
  TestData data{/* A = */ {1, 2, 3, 4},
                /* B = */ {2, 3, 4, 5},
                /* out = */ {2, 6, 12, 20},
                /* shape(A) = */ {2, 2},
                /* shape(B) = */ {2, 2},
                /* shape(out) = */ {2, 2}};

  MulInplaceTestCase testCase;
  testCase.checkBasicOpIsInplaced(data);
}

// Arg0 is larger than arg1
// (2x2) * (1x2)
// Expect that the IR does not contain the basic op and was replaced by the LHS
// inplace variant.
BOOST_AUTO_TEST_CASE(Inplace_mul1) {
  TestData data{/* A = */ {1, 2, 3, 4},
                /* B = */ {2, 3},
                /* out = */ {2, 6, 6, 12},
                /* shape(A) = */ {2, 2},
                /* shape(B) = */ {1, 2},
                /* shape(out) = */ {2, 2}};

  MulInplaceTestCase testCase;
  testCase.checkBasicOpIsLhsInplaced(data);
}

// Arg1 is larger than arg0
// (1x2) * (2x2)
// Expect that the IR does not contain the basic op and was replaced by the RHS
// inplace variant.
BOOST_AUTO_TEST_CASE(Inplace_mul2) {
  TestData data{/* A = */ {2, 3},
                /* B = */ {1, 2, 3, 4},
                /* out = */ {2, 6, 6, 12},
                /* shape(A) = */ {1, 2},
                /* shape(B) = */ {2, 2},
                /* shape(out) = */ {2, 2}};

  MulInplaceTestCase testCase;
  testCase.checkBasicOpIsRhsInplaced(data);
}

// Arg0 and arg1 are of different ranks
// (2x2) * (2)
// Expect that the IR does not contain the basic op and was replaced by the LHS
// inplace variant.
BOOST_AUTO_TEST_CASE(Inplace_mul3) {
  TestData data{/* A = */ {1, 2, 3, 4},
                /* B = */ {2, 3},
                /* out = */ {2, 6, 6, 12},
                /* shape(A) = */ {2, 2},
                /* shape(B) = */ {2},
                /* shape(out) = */ {2, 2}};

  MulInplaceTestCase testCase;
  testCase.checkBasicOpIsLhsInplaced(data);
}

// Arg0 and arg1 are of different ranks
// (2) * (2x2)
// Expect that the IR does not contain the basic op and was replaced by the RHS
// inplace variant.
BOOST_AUTO_TEST_CASE(Inplace_mul4) {
  TestData data{/* A = */ {2, 3},
                /* B = */ {1, 2, 3, 4},
                /* out = */ {2, 6, 6, 12},
                /* shape(A) = */ {2},
                /* shape(B) = */ {2, 2},
                /* shape(out) = */ {2, 2}};

  MulInplaceTestCase testCase;
  testCase.checkBasicOpIsRhsInplaced(data);
}

// Checking MulOp fwdRegMap
BOOST_AUTO_TEST_CASE(Mul_fwdRegMap0) { MulInplaceTestCase().checkFwdRegMap(); }

// Checking MulOp fwdRegMap
BOOST_AUTO_TEST_CASE(Mul_bwdRegMap0) { MulInplaceTestCase().checkBwdRegMap(); }
