// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE BuilderPartialsTest

#include <algorithm>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/test/unit_test.hpp>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <popart/builder.hpp>
#include <popart/error.hpp>
#include <popart/names.hpp>
#include <popart/tensorinfo.hpp>

#include "popart/builder.gen.hpp"

using namespace popart;

struct RunOp {
  std::function<TensorId(Builder &)> _run;
  std::unique_ptr<Builder> builder = Builder::create();

  TensorId operator()() { return _run(*builder); }
};

bool checkErrorMsgIsForUnsupportedPartialsType(const error &ex) {
  // Thrown from Builder::setPartialsType code.
  const auto expected_prefix = "Builder::setPartialsType";

  return boost::algorithm::starts_with(ex.what(), expected_prefix);
}

bool checkErrorMsgIsForNonExistantAttribute(const error &ex) {
  // Thrown from BuilderImpl::getNodeAttribute code.
  const auto expected_prefix =
      "Node does not have an attribute " + std::string(sPartialsTypeAttribute);

  return boost::algorithm::starts_with(ex.what(), expected_prefix);
}

// NB: The builder is not responsible for verifying the value of the
//     partials type attribute. We are only checking it set the attribute
//     as the user directed; we do not check that only certain values are
//     allowed. Malformed partial type values will cause an error in the
//     session (when the Op factory function is called and it tries to
//     parse the attribute).
//
//     On the other hand, we do check that only certain Ops are allowed, such
//     as convolutions.
void testPartialsTypeHandlesSupportedOps() {
  RunOp runMatMul{[](Builder &builder) {
    TensorInfo shape{"FLOAT16", std::vector<int64_t>{4, 4}};

    auto t1 = builder.addInputTensor(shape);
    auto t2 = builder.addInputTensor(shape);

    auto t3 = builder.aiOnnxOpset10().matmul({t1, t2});

    return t3;
  }};

  RunOp runConv{[](Builder &builder) {
    auto t1 =
        builder.addInputTensor({"FLOAT16", std::vector<int64_t>{1, 2, 4, 4}});
    auto k =
        builder.addInputTensor({"FLOAT16", std::vector<int64_t>{2, 2, 3, 3}});

    auto t2 = builder.aiOnnxOpset10().conv(
        {t1, k}, {1, 1}, 1, {}, {1, 1, 1, 1}, {1, 1});

    return t2;
  }};

  RunOp runSupportedOps[] = {std::move(runMatMul), std::move(runConv)};

  // Check the builder can handle the supported ops.
  for (auto &runOp : runSupportedOps) {
    for (const std::string pt : {"FLOAT", "HALF"}) {
      // Run the op and return the output tensor, then set the partials type.
      const auto tOut = runOp();
      auto &builder   = runOp.builder;

      builder->setPartialsType(tOut, pt);

      // Check node attributes directly.
      std::string actual_pt;
      BOOST_REQUIRE_NO_THROW((actual_pt = builder->getStringNodeAttribute(
                                  sPartialsTypeAttribute, {tOut})));
      BOOST_REQUIRE_EQUAL(actual_pt, pt);

      // Now we know set was correct, check get.
      BOOST_REQUIRE_EQUAL(pt, builder->getPartialsType(tOut));
    }
  }
}

void testPartialsTypeHandlesUnsupportedOps() {
  RunOp runAdd{[](Builder &builder) {
    TensorInfo shape{"FLOAT16", std::vector<int64_t>{4, 4, 4}};

    auto t1 = builder.addInputTensor(shape);
    auto t2 = builder.addInputTensor(shape);

    auto t3 = builder.aiOnnxOpset10().add({t1, t2});

    return t3;
  }};

  RunOp runMatMulInt{[](Builder &builder) {
    TensorInfo shape{"INT8", std::vector<int64_t>{4, 4}};

    auto t1 = builder.addInputTensor(shape);
    auto t2 = builder.addInputTensor(shape);

    auto t3 = builder.aiOnnxOpset10().matmulinteger({t1, t2});

    return t3;
  }};

  RunOp runUnsupportedOps[] = {std::move(runAdd), std::move(runMatMulInt)};

  for (auto &runOp : runUnsupportedOps) {
    const auto tOut = runOp();
    auto &builder   = runOp.builder;

    // Check gives default
    BOOST_CHECK_EQUAL(builder->getPartialsType(tOut), "FLOAT");

    // Check throws correctly when trying to set
    BOOST_CHECK_EXCEPTION(builder->setPartialsType(tOut, "FLOAT"),
                          error,
                          checkErrorMsgIsForUnsupportedPartialsType);

    // Check nowhere did the builder erroneously create the attribute, and thus
    // throws correctly.
    BOOST_CHECK_EXCEPTION(
        builder->getStringNodeAttribute(sPartialsTypeAttribute, {tOut}),
        error,
        checkErrorMsgIsForNonExistantAttribute);
  }
}

BOOST_AUTO_TEST_CASE(Builder_PartialsType) {
  testPartialsTypeHandlesSupportedOps();
  testPartialsTypeHandlesUnsupportedOps();
}
