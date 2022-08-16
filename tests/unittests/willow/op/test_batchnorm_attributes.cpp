// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE Test_Willow_Op_Batchnorm_Attribute
#include <boost/test/unit_test.hpp>

#include "popart/operators.hpp"
#include "popart/opserialiser.hpp"
#include "popart/tensorinfo.hpp"
#include <popart/op/batchnorm.hpp>

#include <popart/error.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>

using namespace popart;

/**
 * Test that batchNormOp.appendOutlineAttributes correctly appends
 * the unbiased_variance attribute
 */

BOOST_AUTO_TEST_CASE(TestUnbiasedVarianceAttributeIsAdded) {
  Ir ir;
  Graph &g = ir.getMainGraph();
  class OpSerialiserTester : public OpSerialiserBase {
  public:
    OpSerialiserTester() : OpSerialiserBase(), appended(false) {}

    bool appended;

    void appendAttribute(const std::string &s, float f) {
      OpSerialiserBase::appendAttribute(s, f);
      std::cout << "FLOAT" << std::endl;
    }
    virtual void appendAttribute(const std::string &,
                                 nonstd::optional<int64_t>) override {}
    virtual void appendAttribute(const std::string &,
                                 nonstd::optional<float>) override {}
    virtual void appendAttribute(const std::string &,
                                 nonstd::optional<double>) override {}
    virtual void appendAttribute(const std::string &,
                                 const std::map<TensorId, uint64_t>) override {}
    virtual void appendForwardOp(const Op *) override {}

  private:
    void appendStrAttr(const std::string &s, const std::string &value) final {
      if (s == "unbiased_variance" && value == "true") {
        appended = true;
      }
    }
  };

  BatchNormOp bn_op_unbiased{Onnx::CustomOperators::BatchNormalization_1,
                             1e-5f,
                             0.9f,
                             1,
                             true,
                             Op::Settings(g, "Batchnorm")};

  BatchNormOp bn_op_biased{Onnx::CustomOperators::BatchNormalization_1,
                           1e-5f,
                           0.9f,
                           1,
                           false,
                           Op::Settings(g, "Batchnorm")};

  OpSerialiserTester ost1;
  bn_op_unbiased.appendOutlineAttributes(ost1);
  OpSerialiserTester ost2;
  bn_op_biased.appendOutlineAttributes(ost2);

  BOOST_TEST(ost1.appended == true);
  BOOST_TEST(ost2.appended == false);
}
