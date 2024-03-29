// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE Test_SGD_Optimizer
#include <boost/test/unit_test.hpp>
#include <cstddef>
#include <functional>
#include <initializer_list>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include <popart/compoundscalarhelper.hpp>
#include <popart/ir.hpp>
#include <popart/op/sgd0combo.hpp>
#include <popart/op/sgd1combo.hpp>
#include <popart/op/sgd2combo.hpp>
#include <popart/sessionoptions.hpp>
#include <popart/sgd.hpp>
#include <popart/tensor.hpp>

#include "popart/clipnormsettings.hpp"
#include "popart/datatype.hpp"
#include "popart/op.hpp"
#include "popart/op/sgdcombobase.hpp"
#include "popart/op/varupdate.hpp"
#include "popart/optimizer.hpp"
#include "popart/optimizervalue.hpp"
#include "popart/optimizervaluemap.hpp"
#include "popart/tensordebuginfo.hpp"
#include "popart/tensorinfo.hpp"
#include "popart/tensornames.hpp"
#include "sgdtestcase.hpp"

namespace popart {
class Graph;
class runtime_error;
} // namespace popart

using namespace popart;

BOOST_AUTO_TEST_CASE(TestCtorGivesDefaultHyperParamsEquivalentToUnsetMethods) {
  SGD sgd;

  // All hyper-params should have a single default value.

  BOOST_CHECK(!sgd.hasSpecific());

  const auto test =
      [](const SGD &sgd, auto actualValueMap, auto expectedValue) {
        const OptimizerValueMap &actual = actualValueMap(sgd);
        const OptimizerValueMap expected{expectedValue(sgd)};

        BOOST_CHECK_NO_THROW(actual.validReplacement(expected));
      };

  test(
      sgd,
      [](const SGD &sgd) { return sgd.learningRates(); },
      [](const SGD &sgd) { return sgd.getUnsetLearningRate(); });

  test(
      sgd,
      [](const SGD &sgd) { return sgd.weightDecays(); },
      [](const SGD &sgd) { return sgd.getUnsetWeightDecay(); });

  test(
      sgd,
      [](const SGD &sgd) { return sgd.momentums(); },
      [](const SGD &sgd) { return sgd.getUnsetMomentum(); });

  test(
      sgd,
      [](const SGD &sgd) { return sgd.dampenings(); },
      [](const SGD &sgd) { return sgd.getUnsetDampening(); });

  test(
      sgd,
      [](const SGD &sgd) { return sgd.nesterov(); },
      [](const SGD &sgd) { return sgd.getUnsetNesterov(); });
}

BOOST_AUTO_TEST_CASE(TestCtorValidation) {
  using TestFunc = std::function<void(Graph &)>;

  struct SGD2CtorValidationTestFixture {
    Ir ir;

    void requireNoThrow(TestFunc testF) {
      BOOST_REQUIRE_NO_THROW(testF(ir.getMainGraph()));
    }
  };

  SGD2CtorValidationTestFixture tf;

  tf.requireNoThrow([](Graph &g) {
    SGD{{},
        {},
        SGDAccumulatorAndMomentum::Separate /* default should not throw! */};
  });

  const auto validDataTypes = {
      DataType::UNDEFINED, DataType::FLOAT, DataType::FLOAT16};

  for (const auto dtAccum : validDataTypes) {
    for (const auto dtAccl1 : validDataTypes) {
      tf.requireNoThrow([=](Graph &g) {
        SGD{{}, {}, SGDAccumulatorAndMomentum::Separate, dtAccum, dtAccl1};
      });
    }
  }
}

BOOST_AUTO_TEST_CASE(TestHashAndEq) {
  // Valid hash func: Equal values have the same hash. Note, sgdAccMm and accum
  // & accl1 DataTypes ignored if SGD0.
  SGD sgd0{{},
           {},
           SGDAccumulatorAndMomentum::Separate,
           DataType::STRING,
           DataType::COMPLEX64};
  SGD sgd1{{}, {}, SGDAccumulatorAndMomentum::Combined};
  BOOST_CHECK_NO_THROW(sgd0.validReplacement(sgd1));
  BOOST_CHECK_EQUAL(sgd0.hash(), sgd1.hash());

  // Good hash func: Non-equal values have different hashes.
  SGD sgd2{{{"defaultMomentum", {0.38f, true}}}, // SGD1
           {},
           SGDAccumulatorAndMomentum::Separate,
           DataType::FLOAT,
           DataType::FLOAT};
  BOOST_CHECK_THROW(sgd1.validReplacement(sgd2), runtime_error);
  BOOST_CHECK(sgd1.hash() != sgd2.hash());

  // Test accum and accl1 DataTypes are accounted for too, if
  // SGDAccumulatorAndMomentum::Separate.

  SGD sgd3{{{"defaultMomentum", {0.38f, true}}},
           {},
           SGDAccumulatorAndMomentum::Separate,
           DataType::FLOAT,
           DataType::UNDEFINED};
  BOOST_CHECK_THROW(sgd2.validReplacement(sgd3), runtime_error);
  BOOST_CHECK(sgd2.hash() != sgd3.hash());

  // Cannot replace non-nesterov SGD with nesterov SGD.
  SGD sgd4{{{"defaultMomentum", {0.38f, true}}},
           {},
           SGDAccumulatorAndMomentum::Separate,
           DataType::FLOAT,
           DataType::UNDEFINED};
  BOOST_CHECK_NO_THROW(sgd3.validReplacement(sgd4));
  BOOST_CHECK_EQUAL(sgd3.hash(), sgd4.hash());

  SGD sgd5{{{"defaultMomentum", {0.38f, true}}, {"nesterov", {true, true}}},
           {},
           SGDAccumulatorAndMomentum::Separate,
           DataType::FLOAT,
           DataType::UNDEFINED};
  BOOST_CHECK_THROW(sgd4.validReplacement(sgd5), runtime_error);
  BOOST_CHECK(sgd4.hash() != sgd5.hash());
}

BOOST_AUTO_TEST_CASE(TestGetInputIds_SGD0) {
  SGD0TestCase tc;
  tc.setFactorsFromOptions();

  // Note, inserting one specific value gives the weight all specific values.
  tc.sgd.insertSpecific(tc.wId,
                        std::map<std::string, std::pair<float, bool>>(
                            {{"weightDecay", {0.08f, false}}}));

  const auto inputIds = tc.sgd.getInputIds(*tc.w);

  constexpr size_t expectedNumSgd0Inputs = 4u;

  BOOST_REQUIRE_EQUAL(inputIds.size(), expectedNumSgd0Inputs);
  BOOST_REQUIRE_EQUAL(inputIds[VarUpdateOp::getVarToUpdateInIndex()], tc.wId);
  BOOST_REQUIRE_EQUAL(inputIds[VarUpdateWithUpdaterOp::getUpdaterInIndex()],
                      getGradId(tc.wId));
  // Specific and const.
  BOOST_REQUIRE_EQUAL(inputIds[SGD0ComboOp::getSlr0InIndex()], "");
  // Specific and non-const.
  BOOST_REQUIRE_EQUAL(inputIds[SGD0ComboOp::getWdsf0InIndex()],
                      reservedSpecificWeightDecayScaleFactor0Prefix() + tc.wId);
}

BOOST_AUTO_TEST_CASE(TestGetInputIds_SGD1_2) {

  // Test happens to be exact same for SGD1 and SGD2.
  const auto test = [](auto &tc) {
    tc.setFactorsFromOptions();

    const auto inputIds = tc.sgd.getInputIds(*tc.w);

    bool isNesterov = tc.sgd.nesterov().get(tc.w->id).val();

    size_t expectedNumSgd1or2Inputs = isNesterov ? 10u : 6u;

    BOOST_REQUIRE_EQUAL(inputIds.size(), expectedNumSgd1or2Inputs);
    BOOST_REQUIRE_EQUAL(inputIds[VarUpdateOp::getVarToUpdateInIndex()], tc.wId);
    BOOST_REQUIRE_EQUAL(inputIds[VarUpdateWithUpdaterOp::getUpdaterInIndex()],
                        getGradId(tc.wId));
    BOOST_REQUIRE_EQUAL(inputIds[SGDMComboBaseOp::getSmm1InIndex()], "");
    BOOST_REQUIRE_EQUAL(inputIds[SGDMComboBaseOp::getDpsf1InIndex()], "");
    BOOST_REQUIRE_EQUAL(inputIds[SGDMComboBaseOp::getSwd1InIndex()], "");
    BOOST_REQUIRE_EQUAL(inputIds[SGDMComboBaseOp::getSlr1InIndex()], "");
    if (isNesterov) {
      BOOST_REQUIRE_EQUAL(inputIds[SGDMComboBaseOp::getNdsfInIndex()], "");
      BOOST_REQUIRE_EQUAL(inputIds[SGDMComboBaseOp::getNgsfInIndex()], "");
      BOOST_REQUIRE_EQUAL(inputIds[SGDMComboBaseOp::getMmInIndex()], "");
      BOOST_REQUIRE_EQUAL(inputIds[SGDMComboBaseOp::getWdInIndex()], "");
    }
  };

  {
    SGD1TestCase tc;
    test(tc);
  }
  {
    SGD2TestCase tc;
    test(tc);
  }
  {
    SGD1NesterovTestCase tc;
    test(tc);
  }
  {
    SGD2NesterovTestCase tc;
    test(tc);
  }
}

namespace {

template <typename TC, typename Tester>
void testOptimizerInputs(
    const TC &tc,
    std::vector<std::tuple<popart::TensorId, popart::TensorInfo>> actualInputs,
    const Tester findInputAndRunTests) {
  bool foundInput = false;
  for (const auto &i : actualInputs) {
    const auto &tId   = std::get<0>(i);
    const auto &tInfo = std::get<1>(i);

    foundInput = foundInput || findInputAndRunTests(tc, tId, tInfo);
    if (foundInput) {
      break;
    }
  }
  BOOST_REQUIRE(foundInput);
}

} // namespace

using SGDAllTestCaseTypes = std::tuple<SGD0TestCase,
                                       SGD1TestCase,
                                       SGD2TestCase,
                                       SGD1NesterovTestCase,
                                       SGD2NesterovTestCase>;
BOOST_AUTO_TEST_CASE_TEMPLATE(TestGetOptimizerInputs_Default,
                              SGDTestCaseTy,
                              SGDAllTestCaseTypes) {
  SGDTestCaseTy tc;
  tc.setFactorsFromOptions();

  auto inputs = tc.sgd.getOptimizerInputs(*tc.w);
  BOOST_REQUIRE_EQUAL(inputs.size(), 0u);
}

BOOST_AUTO_TEST_CASE(TestGetOptimizerInputs_SGD0) {
  SGD0TestCase tc;
  tc.setFactorsFromOptions();
  // Give specific non-const values so getOptimizerInputs returns a value for
  // them.
  tc.sgd.insertSpecific(
      tc.wId,
      std::map<std::string, std::pair<float, bool>>(
          {{"weightDecay", {0.08f, false}}, {"learningRate", {0.01f, false}}}));

  auto inputs = tc.sgd.getOptimizerInputs(*tc.w);
  BOOST_REQUIRE_EQUAL(inputs.size(), 2u);

  const auto testWdsf0 =
      [](const auto &tc, const TensorId &tId, const TensorInfo &tInfo) {
        bool matched = false;
        if (tId == reservedSpecificWeightDecayScaleFactor0Prefix() + tc.wId) {
          matched = true;
          BOOST_REQUIRE_EQUAL(tInfo, TensorInfo(tc.w->info.dataType(), {}));
        }
        return matched;
      };
  const auto testSlr0 =
      [](const auto &tc, const TensorId &tId, const TensorInfo &tInfo) {
        bool matched = false;
        if (tId == reservedSpecificScaledLearningRate0Prefix() + tc.wId) {
          matched = true;

          TensorInfo expected{DataType::FLOAT, {}};
          BOOST_REQUIRE_EQUAL(tInfo, expected);
        }
        return matched;
      };

  testOptimizerInputs(tc, inputs, testWdsf0);
  testOptimizerInputs(tc, inputs, testSlr0);
}

BOOST_AUTO_TEST_CASE(TestGetOptimizerInputs_SGD1) {
  SGD1TestCase tc;
  tc.setFactorsFromOptions();
  // Give specific non-const values so getOptimizerInputs returns a value for
  // them.
  tc.sgd.insertSpecific(
      tc.wId,
      std::map<std::string, std::pair<float, bool>>(
          {{"momentum", {0.38f, false}}, {"learningRate", {0.01f, false}}}));

  auto inputs = tc.sgd.getOptimizerInputs(*tc.w);
  BOOST_REQUIRE_EQUAL(inputs.size(), 2u);

  const auto testSmm1 =
      [](const auto &tc, const TensorId &tId, const TensorInfo &tInfo) {
        bool matched = false;
        if (tId == reservedSpecificScaledMomentum1Prefix() + tc.wId) {
          matched = true;
          BOOST_REQUIRE_EQUAL(tInfo, TensorInfo(tc.w->info.dataType(), {}));
        }
        return matched;
      };
  const auto testSlr1 =
      [](const auto &tc, const TensorId &tId, const TensorInfo &tInfo) {
        bool matched = false;
        if (tId == reservedSpecificScaledLearningRate1Prefix() + tc.wId) {
          matched = true;

          TensorInfo expected{DataType::FLOAT, {}};
          BOOST_REQUIRE_EQUAL(tInfo, expected);
        }
        return matched;
      };

  testOptimizerInputs(tc, inputs, testSmm1);
  testOptimizerInputs(tc, inputs, testSlr1);
}

BOOST_AUTO_TEST_CASE(TestGetOptimizerInputs_SGD1Nesterov) {
  SGD1NesterovTestCase tc;
  tc.setFactorsFromOptions();
  // Give specific non-const values so getOptimizerInputs returns a value for
  // them.
  tc.sgd.insertSpecific(tc.wId,
                        std::map<std::string, std::pair<float, bool>>({
                            {"momentum", {0.38f, false}},
                            {"learningRate", {0.01f, false}},
                            {"dampening", {0.0f, false}},
                        }));

  auto inputs = tc.sgd.getOptimizerInputs(*tc.w);
  BOOST_REQUIRE_EQUAL(inputs.size(), 6u);

  const auto testSmm1 =
      [](const auto &tc, const TensorId &tId, const TensorInfo &tInfo) {
        bool matched = false;
        if (tId == reservedSpecificScaledMomentum1Prefix() + tc.wId) {
          matched = true;
          BOOST_REQUIRE_EQUAL(tInfo, TensorInfo(tc.w->info.dataType(), {}));
        }
        return matched;
      };
  const auto testSlr1 =
      [](const auto &tc, const TensorId &tId, const TensorInfo &tInfo) {
        bool matched = false;
        if (tId == reservedSpecificScaledLearningRate1Prefix() + tc.wId) {
          matched = true;

          TensorInfo expected{DataType::FLOAT, {}};
          BOOST_REQUIRE_EQUAL(tInfo, expected);
        }
        return matched;
      };
  const auto testDpsf1 =
      [](const auto &tc, const TensorId &tId, const TensorInfo &tInfo) {
        bool matched = false;
        if (tId == reservedSpecificDampeningScaleFactor1Prefix() + tc.wId) {
          matched = true;

          TensorInfo expected{DataType::FLOAT, {}};
          BOOST_REQUIRE_EQUAL(tInfo, expected);
        }
        return matched;
      };
  const auto testNdsf1 = [](const auto &tc,
                            const TensorId &tId,
                            const TensorInfo &tInfo) {
    bool matched = false;
    if (tId == reservedSpecificNesterovDampeningScaleFactor1Prefix() + tc.wId) {
      matched = true;

      TensorInfo expected{DataType::FLOAT, {}};
      BOOST_REQUIRE_EQUAL(tInfo, expected);
    }
    return matched;
  };
  const auto testMm =
      [](const auto &tc, const TensorId &tId, const TensorInfo &tInfo) {
        bool matched = false;
        if (tId == reservedSpecificSGDMomentumPrefix() + tc.wId) {
          matched = true;

          TensorInfo expected{DataType::FLOAT, {}};
          BOOST_REQUIRE_EQUAL(tInfo, expected);
        }
        return matched;
      };

  testOptimizerInputs(tc, inputs, testSmm1);
  testOptimizerInputs(tc, inputs, testSlr1);
  testOptimizerInputs(tc, inputs, testDpsf1);
  // Inputs for nesterov momentum
  testOptimizerInputs(tc, inputs, testNdsf1);
  testOptimizerInputs(tc, inputs, testMm);
}

BOOST_AUTO_TEST_CASE(TestGetOptimizerInputs_SGD2) {
  SGD2TestCase tc;
  tc.setFactorsFromOptions();
  // Give specific non-const values so getOptimizerInputs returns a value for
  // them.
  tc.sgd.insertSpecific(
      tc.wId,
      std::map<std::string, std::pair<float, bool>>(
          {{"momentum", {0.38f, false}}, {"learningRate", {0.01f, false}}}));

  auto inputs = tc.sgd.getOptimizerInputs(*tc.w);
  BOOST_REQUIRE_EQUAL(inputs.size(), 2u);

  const auto testSmm1 =
      [](const auto &tc, const TensorId &tId, const TensorInfo &tInfo) {
        bool matched = false;
        if (tId == reservedSpecificScaledMomentum2Prefix() + tc.wId) {
          matched = true;
          BOOST_REQUIRE_EQUAL(tInfo, TensorInfo(tc.w->info.dataType(), {}));
        }
        return matched;
      };
  const auto testSlr1 =
      [](const auto &tc, const TensorId &tId, const TensorInfo &tInfo) {
        bool matched = false;
        if (tId == reservedSpecificScaledLearningRate2Prefix() + tc.wId) {
          matched = true;

          TensorInfo expected{DataType::FLOAT, {}};
          BOOST_REQUIRE_EQUAL(tInfo, expected);
        }
        return matched;
      };

  testOptimizerInputs(tc, inputs, testSmm1);
  testOptimizerInputs(tc, inputs, testSlr1);
}

BOOST_AUTO_TEST_CASE(TestGetOptimizerInputs_SGD2Nesterov) {
  SGD2NesterovTestCase tc;
  tc.setFactorsFromOptions();
  // Give specific non-const values so getOptimizerInputs returns a value for
  // them.
  tc.sgd.insertSpecific(tc.wId,
                        std::map<std::string, std::pair<float, bool>>({
                            {"momentum", {0.38f, false}},
                            {"learningRate", {0.01f, false}},
                            {"dampening", {0.0f, false}},
                        }));

  auto inputs = tc.sgd.getOptimizerInputs(*tc.w);
  BOOST_REQUIRE_EQUAL(inputs.size(), 6u);

  const auto testSmm2 =
      [](const auto &tc, const TensorId &tId, const TensorInfo &tInfo) {
        bool matched = false;
        if (tId == reservedSpecificScaledMomentum2Prefix() + tc.wId) {
          matched = true;
          BOOST_REQUIRE_EQUAL(tInfo, TensorInfo(tc.w->info.dataType(), {}));
        }
        return matched;
      };
  const auto testSlr2 =
      [](const auto &tc, const TensorId &tId, const TensorInfo &tInfo) {
        bool matched = false;
        if (tId == reservedSpecificScaledLearningRate2Prefix() + tc.wId) {
          matched = true;

          TensorInfo expected{DataType::FLOAT, {}};
          BOOST_REQUIRE_EQUAL(tInfo, expected);
        }
        return matched;
      };
  const auto testDpsf2 =
      [](const auto &tc, const TensorId &tId, const TensorInfo &tInfo) {
        bool matched = false;
        if (tId == reservedSpecificDampeningScaleFactor2Prefix() + tc.wId) {
          matched = true;

          TensorInfo expected{DataType::FLOAT, {}};
          BOOST_REQUIRE_EQUAL(tInfo, expected);
        }
        return matched;
      };
  const auto testNdsf2 = [](const auto &tc,
                            const TensorId &tId,
                            const TensorInfo &tInfo) {
    bool matched = false;
    if (tId == reservedSpecificNesterovDampeningScaleFactor2Prefix() + tc.wId) {
      matched = true;

      TensorInfo expected{DataType::FLOAT, {}};
      BOOST_REQUIRE_EQUAL(tInfo, expected);
    }
    return matched;
  };
  const auto testMm =
      [](const auto &tc, const TensorId &tId, const TensorInfo &tInfo) {
        bool matched = false;
        if (tId == reservedSpecificSGDMomentumPrefix() + tc.wId) {
          matched = true;

          TensorInfo expected{DataType::FLOAT, {}};
          BOOST_REQUIRE_EQUAL(tInfo, expected);
        }
        return matched;
      };

  testOptimizerInputs(tc, inputs, testSmm2);
  testOptimizerInputs(tc, inputs, testSlr2);
  testOptimizerInputs(tc, inputs, testDpsf2);
  // Inputs for nesterov momentum
  testOptimizerInputs(tc, inputs, testNdsf2);
  testOptimizerInputs(tc, inputs, testMm);
}

/* The following CreateOp tests are integration test with the
 * CompoundScalarHelpers. They test that createOp creates a VarUpdateOp whose
 * OptimizerValues are what the helpers say they should be, as opposed to
 * directly checking for the correct value. There are other tests that assert
 * the CompoundScalarHelpers are numerically correct. */

BOOST_AUTO_TEST_CASE(TestCreateOpIntegrationWithCompoundScalarHelper_SGD0) {
  SGD0TestCase tc;
  tc.setFactorsFromOptions();

  const auto op = tc.sgd.createOp(*tc.w, tc.graph());

  const auto sgd0 = dynamic_cast<const SGD0ComboOp *>(op.get());
  BOOST_REQUIRE(sgd0);
  BOOST_REQUIRE(sgd0->initSlr0 ==
                ScaledLearningRate0Helper{}.getFromWeightId(tc.wId, tc.sgd));
  BOOST_REQUIRE(
      sgd0->initWdsf0 ==
      WeightDecayScaleFactor0Helper{}.getFromWeightId(tc.wId, tc.sgd));
}

BOOST_AUTO_TEST_CASE(TestCreateOpIntegrationWithCompoundScalarHelper_SGD0_GA) {
  SGD0TestCase tc;
  SessionOptions opts;
  opts.enableGradientAccumulation = true;
  opts.accumulationFactor         = 2;
  tc.ir.setUserOptions(opts);
  tc.setFactorsFromOptions();

  const auto op = tc.sgd.createOp(*tc.w, tc.graph());

  const auto sgd0 = dynamic_cast<const SGD0ComboOp *>(op.get());
  BOOST_REQUIRE(sgd0);
  BOOST_REQUIRE(sgd0->initSlr0 ==
                ScaledLearningRate0Helper{}.getFromWeightId(tc.wId, tc.sgd));
  BOOST_REQUIRE(
      sgd0->initWdsf0 ==
      WeightDecayScaleFactor0Helper{}.getFromWeightId(tc.wId, tc.sgd));
}

BOOST_AUTO_TEST_CASE(TestCreateOpIntegrationWithCompoundScalarHelper_SGD1) {
  {
    SGD1TestCase tc;
    tc.setFactorsFromOptions();

    const auto op = tc.sgd.createOp(*tc.w, tc.graph());

    const auto sgd1 = dynamic_cast<const SGD1ComboOp *>(op.get());
    BOOST_REQUIRE(sgd1);

    BOOST_REQUIRE(sgd1->initSmm1 ==
                  ScaledMomentum1Helper{}.getFromWeightId(tc.wId, tc.sgd));
    BOOST_REQUIRE(
        sgd1->initDpsf1 ==
        DampeningScaleFactor1Helper{}.getFromWeightId(tc.wId, tc.sgd));
    BOOST_REQUIRE(sgd1->initSwd1 ==
                  ScaledWeightDecay1Helper{}.getFromWeightId(tc.wId, tc.sgd));
    BOOST_REQUIRE(sgd1->initSlr1 ==
                  ScaledLearningRate1Helper{}.getFromWeightId(tc.wId, tc.sgd));
  }
  {
    SGD1NesterovTestCase tc;
    tc.setFactorsFromOptions();

    const auto op = tc.sgd.createOp(*tc.w, tc.graph());

    const auto sgd1 = dynamic_cast<const SGD1ComboOp *>(op.get());
    BOOST_REQUIRE(sgd1);

    BOOST_REQUIRE(sgd1->initSmm1 ==
                  ScaledMomentum1Helper{}.getFromWeightId(tc.wId, tc.sgd));
    BOOST_REQUIRE(
        sgd1->initDpsf1 ==
        DampeningScaleFactor1Helper{}.getFromWeightId(tc.wId, tc.sgd));
    BOOST_REQUIRE(sgd1->initSwd1 ==
                  ScaledWeightDecay1Helper{}.getFromWeightId(tc.wId, tc.sgd));
    BOOST_REQUIRE(sgd1->initSlr1 ==
                  ScaledLearningRate1Helper{}.getFromWeightId(tc.wId, tc.sgd));
    BOOST_REQUIRE(sgd1->initMm ==
                  SGDMomentumHelper{}.getFromWeightId(tc.wId, tc.sgd));
    BOOST_REQUIRE(sgd1->initWd ==
                  SGDWeightDecayHelper{}.getFromWeightId(tc.wId, tc.sgd));
    BOOST_REQUIRE(
        sgd1->initNgsf ==
        NesterovGradScaleFactor1Helper{}.getFromWeightId(tc.wId, tc.sgd));
    BOOST_REQUIRE(
        sgd1->initNdsf ==
        NesterovDampeningScaleFactor1Helper{}.getFromWeightId(tc.wId, tc.sgd));
  }
}

BOOST_AUTO_TEST_CASE(TestCreateOpIntegrationWithCompoundScalarHelper_SGD2) {
  {
    SGD2TestCase tc;
    tc.setFactorsFromOptions();

    const auto op = tc.sgd.createOp(*tc.w, tc.graph());

    const auto sgd2 = dynamic_cast<const SGD2ComboOp *>(op.get());
    BOOST_REQUIRE(sgd2);

    BOOST_REQUIRE(sgd2->initSmm1 ==
                  ScaledMomentum1Helper{}.getFromWeightId(tc.wId, tc.sgd));
    BOOST_REQUIRE(
        sgd2->initDpsf1 ==
        DampeningScaleFactor1Helper{}.getFromWeightId(tc.wId, tc.sgd));
    BOOST_REQUIRE(sgd2->initSwd1 ==
                  ScaledWeightDecay1Helper{}.getFromWeightId(tc.wId, tc.sgd));
    BOOST_REQUIRE(sgd2->initSlr1 ==
                  ScaledLearningRate1Helper{}.getFromWeightId(tc.wId, tc.sgd));
  }
  {
    SGD2NesterovTestCase tc;
    tc.setFactorsFromOptions();

    const auto op = tc.sgd.createOp(*tc.w, tc.graph());

    const auto sgd2 = dynamic_cast<const SGD2ComboOp *>(op.get());
    BOOST_REQUIRE(sgd2);

    BOOST_REQUIRE(sgd2->initSmm1 ==
                  ScaledMomentum1Helper{}.getFromWeightId(tc.wId, tc.sgd));
    BOOST_REQUIRE(
        sgd2->initDpsf1 ==
        DampeningScaleFactor1Helper{}.getFromWeightId(tc.wId, tc.sgd));
    BOOST_REQUIRE(sgd2->initSwd1 ==
                  ScaledWeightDecay1Helper{}.getFromWeightId(tc.wId, tc.sgd));
    BOOST_REQUIRE(sgd2->initSlr1 ==
                  ScaledLearningRate1Helper{}.getFromWeightId(tc.wId, tc.sgd));
    BOOST_REQUIRE(sgd2->initMm ==
                  SGDMomentumHelper{}.getFromWeightId(tc.wId, tc.sgd));
    BOOST_REQUIRE(sgd2->initWd ==
                  SGDWeightDecayHelper{}.getFromWeightId(tc.wId, tc.sgd));
    BOOST_REQUIRE(
        sgd2->initNgsf ==
        NesterovGradScaleFactor2Helper{}.getFromWeightId(tc.wId, tc.sgd));
    BOOST_REQUIRE(
        sgd2->initNdsf ==
        NesterovDampeningScaleFactor2Helper{}.getFromWeightId(tc.wId, tc.sgd));
  }
}

BOOST_AUTO_TEST_CASE(TestCreateOpAllOtherFields_SGD0) {
  {
    // No GradAccum, or Replica
    SGD0TestCase tc;

    SessionOptions opts;
    opts.enableGradientAccumulation        = false;
    opts.enableReplicatedGraphs            = false;
    opts.enableDistributedReplicatedGraphs = false;
    tc.ir.setUserOptions(opts);
    tc.setFactorsFromOptions();

    const auto op = tc.sgd.createOp(*tc.w, tc.graph());

    const auto sgd0 = dynamic_cast<const SGD0ComboOp *>(op.get());
    BOOST_REQUIRE(sgd0);

    BOOST_REQUIRE(sgd0->reductionType == OptimizerReductionType::None);
  }
  {
    // Replica Only
    SGD0TestCase tc;

    SessionOptions opts;
    opts.enableGradientAccumulation = false;
    opts.enableReplicatedGraphs     = true;
    opts.replicatedGraphCount       = 2;
    tc.ir.setUserOptions(opts);
    tc.setFactorsFromOptions();

    const auto op = tc.sgd.createOp(*tc.w, tc.graph());

    const auto sgd0 = dynamic_cast<const SGD0ComboOp *>(op.get());
    BOOST_REQUIRE(sgd0);

    BOOST_REQUIRE(sgd0->reductionType == OptimizerReductionType::GradReduce);
  }
  {
    // GradAccum and Replica
    SGD0TestCase tc;

    SessionOptions opts;
    opts.enableGradientAccumulation = true;
    opts.accumulationFactor         = 2;
    opts.enableReplicatedGraphs     = true;
    opts.replicatedGraphCount       = 2;
    tc.ir.setUserOptions(opts);
    tc.setFactorsFromOptions();

    const auto op = tc.sgd.createOp(*tc.w, tc.graph());

    const auto sgd0 = dynamic_cast<const SGD0ComboOp *>(op.get());
    BOOST_REQUIRE(sgd0);

    BOOST_REQUIRE(sgd0->reductionType == OptimizerReductionType::AccumReduce);
  }
}

BOOST_AUTO_TEST_CASE(TestCreateOpAllOtherFields_SGD1) {
  {
    // Cannot use SGD1TestCase helper for no grad acc case, as it uses no
    // momentum.
    SGDCustomTestCase tc{SGD{{{"defaultMomentum", {0.1, true}}},
                             {},
                             SGDAccumulatorAndMomentum::Combined}};

    SessionOptions opts;
    opts.enableGradientAccumulation        = false;
    opts.enableReplicatedGraphs            = false;
    opts.enableDistributedReplicatedGraphs = false;
    tc.ir.setUserOptions(opts);
    tc.setFactorsFromOptions();

    const auto op = tc.sgd.createOp(*tc.w, tc.graph());

    const auto sgd1 = dynamic_cast<const SGD1ComboOp *>(op.get());
    BOOST_REQUIRE(sgd1);

    BOOST_REQUIRE(sgd1->reductionType == OptimizerReductionType::None);
  }
  {
    SGDCustomTestCase tc{
        SGD{{{"defaultMomentum", {0.1, true}}, {"nesterov", {true, true}}},
            {},
            SGDAccumulatorAndMomentum::Combined}};

    SessionOptions opts;
    opts.enableGradientAccumulation        = false;
    opts.enableReplicatedGraphs            = false;
    opts.enableDistributedReplicatedGraphs = false;
    tc.ir.setUserOptions(opts);
    tc.setFactorsFromOptions();

    const auto op = tc.sgd.createOp(*tc.w, tc.graph());

    const auto sgd1 = dynamic_cast<const SGD1ComboOp *>(op.get());
    BOOST_REQUIRE(sgd1);

    BOOST_REQUIRE(sgd1->reductionType == OptimizerReductionType::None);
  }
  {
    SGD1TestCase tc;

    SessionOptions opts;
    opts.enableGradientAccumulation = true;
    opts.accumulationFactor         = 4;
    opts.enableReplicatedGraphs     = true;
    opts.replicatedGraphCount       = 2;
    tc.ir.setUserOptions(opts);
    tc.setFactorsFromOptions();

    const auto op = tc.sgd.createOp(*tc.w, tc.graph());

    const auto sgd1 = dynamic_cast<const SGD1ComboOp *>(op.get());
    BOOST_REQUIRE(sgd1);

    BOOST_REQUIRE(sgd1->reductionType == OptimizerReductionType::AcclReduce);
  }
  {
    SGD1NesterovTestCase tc;

    SessionOptions opts;
    opts.enableGradientAccumulation = true;
    opts.accumulationFactor         = 4;
    opts.enableReplicatedGraphs     = true;
    opts.replicatedGraphCount       = 2;
    tc.ir.setUserOptions(opts);
    tc.setFactorsFromOptions();

    const auto op = tc.sgd.createOp(*tc.w, tc.graph());

    const auto sgd1 = dynamic_cast<const SGD1ComboOp *>(op.get());
    BOOST_REQUIRE(sgd1);

    BOOST_REQUIRE(sgd1->reductionType == OptimizerReductionType::AcclReduce);
  }
  {
    SGDCustomTestCase tc{SGD{{{"defaultMomentum", {0.1, true}}},
                             {},
                             SGDAccumulatorAndMomentum::Combined}};

    SessionOptions opts;
    opts.enableGradientAccumulation = false;
    opts.enableReplicatedGraphs     = true;
    opts.replicatedGraphCount       = 2;
    tc.ir.setUserOptions(opts);
    tc.setFactorsFromOptions();

    const auto op = tc.sgd.createOp(*tc.w, tc.graph());

    const auto sgd1 = dynamic_cast<const SGD1ComboOp *>(op.get());
    BOOST_REQUIRE(sgd1);

    BOOST_REQUIRE(sgd1->reductionType == OptimizerReductionType::GradReduce);
  }
  {
    SGDCustomTestCase tc{
        SGD{{{"defaultMomentum", {0.1, true}}, {"nesterov", {true, true}}},
            {},
            SGDAccumulatorAndMomentum::Combined}};

    SessionOptions opts;
    opts.enableGradientAccumulation = false;
    opts.enableReplicatedGraphs     = true;
    opts.replicatedGraphCount       = 2;
    tc.ir.setUserOptions(opts);
    tc.setFactorsFromOptions();

    const auto op = tc.sgd.createOp(*tc.w, tc.graph());

    const auto sgd1 = dynamic_cast<const SGD1ComboOp *>(op.get());
    BOOST_REQUIRE(sgd1);

    BOOST_REQUIRE(sgd1->reductionType == OptimizerReductionType::GradReduce);
  }
}

BOOST_AUTO_TEST_CASE(TestCreateOpAllOtherFields_SGD2) {
  {
    // Cannot use SGD1TestCase helper for no grad acc case, as it uses no
    // momentum.
    SGDCustomTestCase tc{SGD{{{"defaultMomentum", {0.1, true}}},
                             {},
                             SGDAccumulatorAndMomentum::Separate,
                             DataType::UNDEFINED,
                             DataType::FLOAT16}};

    SessionOptions opts;
    opts.enableGradientAccumulation        = false;
    opts.enableReplicatedGraphs            = false;
    opts.enableDistributedReplicatedGraphs = false;
    tc.ir.setUserOptions(opts);
    tc.setFactorsFromOptions();

    const auto op = tc.sgd.createOp(*tc.w, tc.graph());

    const auto sgd2 = dynamic_cast<const SGD2ComboOp *>(op.get());
    BOOST_REQUIRE(sgd2);

    BOOST_REQUIRE(sgd2->reductionType == OptimizerReductionType::None);
    BOOST_REQUIRE(!sgd2->withGradAccum);
    BOOST_REQUIRE(sgd2->accumType == tc.w->info.dataType());
    BOOST_REQUIRE(sgd2->accl1Type == DataType::FLOAT16);
  }
  {
    SGDCustomTestCase tc{
        SGD{{{"defaultMomentum", {0.1, true}}, {"nesterov", {true, true}}},
            {},
            SGDAccumulatorAndMomentum::Separate,
            DataType::UNDEFINED,
            DataType::FLOAT16}};

    SessionOptions opts;
    opts.enableGradientAccumulation        = false;
    opts.enableReplicatedGraphs            = false;
    opts.enableDistributedReplicatedGraphs = false;
    tc.ir.setUserOptions(opts);
    tc.setFactorsFromOptions();

    const auto op = tc.sgd.createOp(*tc.w, tc.graph());

    const auto sgd2 = dynamic_cast<const SGD2ComboOp *>(op.get());
    BOOST_REQUIRE(sgd2);

    BOOST_REQUIRE(sgd2->reductionType == OptimizerReductionType::None);
    BOOST_REQUIRE(!sgd2->withGradAccum);
    BOOST_REQUIRE(sgd2->accumType == tc.w->info.dataType());
    BOOST_REQUIRE(sgd2->accl1Type == DataType::FLOAT16);
  }
  {
    SGD2TestCase tc;

    SessionOptions opts;
    opts.enableGradientAccumulation = true;
    opts.accumulationFactor         = 4;
    opts.enableReplicatedGraphs     = true;
    opts.replicatedGraphCount       = 2;
    tc.ir.setUserOptions(opts);
    tc.setFactorsFromOptions();

    const auto op = tc.sgd.createOp(*tc.w, tc.graph());

    const auto sgd2 = dynamic_cast<const SGD2ComboOp *>(op.get());
    BOOST_REQUIRE(sgd2);

    BOOST_REQUIRE(sgd2->reductionType == OptimizerReductionType::AccumReduce);
    BOOST_REQUIRE(sgd2->withGradAccum);
    // SGD2TestCase gives default accumType and accl1Type. Check this was
    // UNDEFINED.
    const auto wDataType = tc.w->info.dataType();
    BOOST_REQUIRE(sgd2->accumType == wDataType);
    BOOST_REQUIRE(sgd2->accl1Type == wDataType);
  }
  {
    SGD2NesterovTestCase tc;

    SessionOptions opts;
    opts.enableGradientAccumulation = true;
    opts.accumulationFactor         = 4;
    opts.enableReplicatedGraphs     = true;
    opts.replicatedGraphCount       = 2;
    tc.ir.setUserOptions(opts);
    tc.setFactorsFromOptions();

    const auto op = tc.sgd.createOp(*tc.w, tc.graph());

    const auto sgd2 = dynamic_cast<const SGD2ComboOp *>(op.get());
    BOOST_REQUIRE(sgd2);

    BOOST_REQUIRE(sgd2->reductionType == OptimizerReductionType::AccumReduce);
    BOOST_REQUIRE(sgd2->withGradAccum);
    // SGD2TestCase gives default accumType and accl1Type. Check this was
    // UNDEFINED.
    const auto wDataType = tc.w->info.dataType();
    BOOST_REQUIRE(sgd2->accumType == wDataType);
    BOOST_REQUIRE(sgd2->accl1Type == wDataType);
  }
  {
    SGDCustomTestCase tc{SGD{{{"defaultMomentum", {0.1, true}}},
                             {},
                             SGDAccumulatorAndMomentum::Separate,
                             DataType::FLOAT16,
                             DataType::FLOAT}};

    SessionOptions opts;
    opts.enableGradientAccumulation = false;
    opts.enableReplicatedGraphs     = true;
    opts.replicatedGraphCount       = 2;
    tc.ir.setUserOptions(opts);
    tc.setFactorsFromOptions();

    const auto op = tc.sgd.createOp(*tc.w, tc.graph());

    const auto sgd2 = dynamic_cast<const SGD2ComboOp *>(op.get());
    BOOST_REQUIRE(sgd2);

    BOOST_REQUIRE(sgd2->reductionType == OptimizerReductionType::GradReduce);
    BOOST_REQUIRE(!sgd2->withGradAccum);
    BOOST_REQUIRE(sgd2->accumType == DataType::FLOAT16);
    BOOST_REQUIRE(sgd2->accl1Type == DataType::FLOAT);
  }
  {
    SGDCustomTestCase tc{
        SGD{{{"defaultMomentum", {0.2, true}}, {"nesterov", {true, true}}},
            {},
            SGDAccumulatorAndMomentum::Separate,
            DataType::FLOAT16,
            DataType::FLOAT}};

    SessionOptions opts;
    opts.enableGradientAccumulation = false;
    opts.enableReplicatedGraphs     = true;
    opts.replicatedGraphCount       = 2;
    tc.ir.setUserOptions(opts);
    tc.setFactorsFromOptions();

    const auto op = tc.sgd.createOp(*tc.w, tc.graph());

    const auto sgd2 = dynamic_cast<const SGD2ComboOp *>(op.get());
    BOOST_REQUIRE(sgd2);

    BOOST_REQUIRE(sgd2->reductionType == OptimizerReductionType::GradReduce);
    BOOST_REQUIRE(!sgd2->withGradAccum);
    BOOST_REQUIRE(sgd2->accumType == DataType::FLOAT16);
    BOOST_REQUIRE(sgd2->accl1Type == DataType::FLOAT);
  }
}
