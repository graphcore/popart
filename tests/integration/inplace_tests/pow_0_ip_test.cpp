// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE Pow0InplaceTest

#include "elementwisebinary_testcase.hpp"
#include <popart/op/pow.hpp>

using namespace popart;

// Test case class for testing inplacing of PowOp
class PowInplaceTestCase : public ElementWiseBinaryTestCase {
public:
  OperatorIdentifier basicOp() const final { return Onnx::AiOnnx::OpSet9::Pow; }

  bool hasLhsOp() const final { return true; }

  OperatorIdentifier lhsOp() const final {
    return Onnx::CustomOperators::PowLhsInplace;
  }

  TensorId insertOp(AiOnnxOpset9 &opset,
                    const TensorId &a,
                    const TensorId &b) const final {
    return opset.pow({a, b});
  }
};

// Basic case where arg0 and arg1 have matching shape
// (2x2) * (2x2)
// Expect that the IR does not contain the basic op and was replaced by the LHS
// inplace variant.
BOOST_AUTO_TEST_CASE(Inplace_pow0) {
  TestData data{/* A = */ {1, 2, 3, 4},
                /* B = */ {2, 3, 4, 5},
                /* out = */ {1, 8, 81, 1024},
                /* shape(A) = */ {2, 2},
                /* shape(B) = */ {2, 2},
                /* shape(out) = */ {2, 2}};

  PowInplaceTestCase testCase;
  testCase.checkBasicOpIsLhsInplaced(data);
}

// Arg0 is larger than arg1
// (2x2) * (1x2)
// Expect that the IR does not contain the basic op and was replaced by the LHS
// inplace variant.
BOOST_AUTO_TEST_CASE(Inplace_pow1) {
  TestData data{/* A = */ {1, 2, 3, 4},
                /* B = */ {2, 3},
                /* out = */ {1, 8, 9, 64},
                /* shape(A) = */ {2, 2},
                /* shape(B) = */ {1, 2},
                /* shape(out) = */ {2, 2}};

  PowInplaceTestCase testCase;
  testCase.checkBasicOpIsLhsInplaced(data);
}

// Arg1 is larger than arg0
// (1x2) * (2x2)
// Expect that the IR contains the basic op (no inplacing applied)
BOOST_AUTO_TEST_CASE(Inplace_pow2) {
  TestData data{/* A = */ {2, 3},
                /* B = */ {1, 2, 3, 4},
                /* out = */ {2, 9, 8, 81},
                /* shape(A) = */ {1, 2},
                /* shape(B) = */ {2, 2},
                /* shape(out) = */ {2, 2}};

  PowInplaceTestCase testCase;
  testCase.checkBasicOpIsNotInplaced(data);
}

// Arg0 and arg1 are of different ranks
// (2x2) * (2)
// Expect that the IR does not contain the basic op and was replaced by the LHS
// inplace variant.
BOOST_AUTO_TEST_CASE(Inplace_pow3) {
  TestData data{/* A = */ {1, 2, 3, 4},
                /* B = */ {2, 3},
                /* out = */ {1, 8, 9, 64},
                /* shape(A) = */ {2, 2},
                /* shape(B) = */ {2},
                /* shape(out) = */ {2, 2}};

  PowInplaceTestCase testCase;
  testCase.checkBasicOpIsLhsInplaced(data);
}

// Arg0 and arg1 are of different ranks
// (2) * (2x2)
// Expect that the IR does contains the basic op (no inplacing applied)
BOOST_AUTO_TEST_CASE(Inplace_pow4) {
  TestData data{/* A = */ {2, 3},
                /* B = */ {1, 2, 3, 4},
                /* out = */ {2, 9, 8, 81},
                /* shape(A) = */ {2},
                /* shape(B) = */ {2, 2},
                /* shape(out) = */ {2, 2}};

  PowInplaceTestCase testCase;
  testCase.checkBasicOpIsNotInplaced(data);
}

// Checking powOp fwdRegMap
BOOST_AUTO_TEST_CASE(Pow_fwdRegMap0) { PowInplaceTestCase().checkFwdRegMap(); }

// Checking powOp fwdRegMap
BOOST_AUTO_TEST_CASE(Pow_bwdRegMap0) { PowInplaceTestCase().checkBwdRegMap(); }
