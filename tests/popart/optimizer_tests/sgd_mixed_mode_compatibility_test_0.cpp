#define BOOST_TEST_MODULE SgdMixedModeCompatTest0

#include <boost/test/unit_test.hpp>
#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/filereader.hpp>
#include <popart/half.hpp>
#include <popart/inputshapeinfo.hpp>
#include <popart/ir.hpp>
#include <popart/names.hpp>
#include <popart/op/l1.hpp>
#include <popart/optimizer.hpp>
#include <popart/tensor.hpp>
#include <popart/tensordata.hpp>
#include <popart/tensors.hpp>

using namespace popart;

BOOST_AUTO_TEST_CASE(ConstExprTest_AddCastMatMul) {

  OptimizerValue globalWeightDecay{1, false};
  OptimizerValue globalLearningRate{1, false};
  OptimizerValue lossScaling{1, false};
  auto opt0 = SGD(globalLearningRate, globalWeightDecay, lossScaling);

  // exactly the same, compatible
  BOOST_CHECK(opt0.validReplacement(SGD(opt0)));
  BOOST_CHECK(opt0.validReplacement(opt0));

  // also all non-const, compatible
  BOOST_CHECK(opt0.validReplacement(SGD({2, false}, {2, false}, {2, false})));

  // any replacement of non-const with const, not compatible
  // applying the diagonal of truth
  BOOST_CHECK(!opt0.validReplacement(SGD({1, true}, {1, false}, {1, false})));
  BOOST_CHECK(!opt0.validReplacement(SGD({1, false}, {1, true}, {1, false})));
  BOOST_CHECK(!opt0.validReplacement(SGD({1, false}, {1, false}, {1, true})));

  // cannot insert a tensor specific value
  auto opt1 = opt0;
  opt1.insertSpecific("foo", {1, false}, {1, false});
  BOOST_CHECK(!opt0.validReplacement(opt1));

  // cannot remove a tensor specific value
  BOOST_CHECK(!opt1.validReplacement(opt0));

  // all const ops
  auto opt2 = SGD({1, true}, {1, true}, {1, true});

  // cannot change value for constant optimizer
  BOOST_CHECK(!opt2.validReplacement(SGD({2, true}, {1, true}, {1, true})));
  BOOST_CHECK(!opt2.validReplacement(SGD({1, true}, {2, true}, {1, true})));
  BOOST_CHECK(!opt2.validReplacement(SGD({1, true}, {1, true}, {2, true})));

  //  Tensor foo has lr 2 and wd 2
  auto opt3 = SGD({1, true}, {1, true}, {1, true});
  opt3.insertSpecific("foo", {2, true}, {2, true});

  //  Tensor foo has lr 3 and wd 3
  auto opt4 = SGD({1, true}, {1, true}, {1, true});
  opt4.insertSpecific("foo", {3, true}, {3, true});

  // Cannot change specific value if it is const
  BOOST_CHECK(!opt3.validReplacement(opt4));
  BOOST_CHECK(!opt4.validReplacement(opt3));
}
