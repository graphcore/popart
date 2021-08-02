// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE SgdMixedModeCompatTest0

#include <boost/test/unit_test.hpp>
#include <filereader.hpp>
#include <iostream>
#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/half.hpp>
#include <popart/inputshapeinfo.hpp>
#include <popart/ir.hpp>
#include <popart/names.hpp>
#include <popart/op/l1.hpp>
#include <popart/sgd.hpp>
#include <popart/tensor.hpp>
#include <popart/tensordata.hpp>
#include <popart/tensors.hpp>

using namespace popart;

BOOST_AUTO_TEST_CASE(SGDMixedModeCompatTest0) {

  SGD opt0({{"defaultLearningRate", {1, false}},
            {"defaultWeightDecay", {1, false}},
            {"lossScaling", {1, false}}});

  // exactly the same, compatible
  std::cout << "Can be replaced by self?" << std::endl;
  opt0.validReplacement(SGD(opt0));
  opt0.validReplacement(opt0);

  // also all non-const, compatible
  std::cout << "All non-consts can be replaced all non-consts?" << std::endl;
  opt0.validReplacement(SGD({{"defaultLearningRate", {2, false}},
                             {"defaultWeightDecay", {2, false}},
                             {"lossScaling", {2, false}}}));

  // any replacement of non-const with const, not compatible
  // applying the diagonal of truth
  std::cout << "All non-consts can be replaced by const lr?" << std::endl;
  BOOST_CHECK_THROW(
      opt0.validReplacement(SGD({{"defaultLearningRate", {1, true}},
                                 {"defaultWeightDecay", {1, false}},
                                 {"lossScaling", {1, false}}})),
      popart::error);

  std::cout << "All non-consts can be replaced by const wd?" << std::endl;
  BOOST_CHECK_THROW(
      opt0.validReplacement(SGD({{"defaultLearningRate", {1, false}},
                                 {"defaultWeightDecay", {1, true}},
                                 {"lossScaling", {1, false}}})),
      popart::error);

  std::cout << "All non-consts can be replaced by const ls?" << std::endl;
  BOOST_CHECK_THROW(
      opt0.validReplacement(SGD({{"defaultLearningRate", {1, false}},
                                 {"defaultWeightDecay", {1, false}},
                                 {"lossScaling", {1, true}}})),
      popart::error);

  // cannot insert a tensor specific value
  auto opt1 = opt0;
  opt1.insertSpecific(
      "foo", {{"learningRate", {1, false}}, {"weightDecay", {1, false}}});
  BOOST_CHECK_THROW(opt0.validReplacement(opt1), popart::error);

  // cannot remove a tensor specific value
  BOOST_CHECK_THROW(opt1.validReplacement(opt0), popart::error);

  // all const optimizer values
  SGD opt2({{"defaultLearningRate", {1, true}},
            {"defaultWeightDecay", {1, true}},
            {"lossScaling", {1, true}}});

  // cannot change value for constant optimizer
  BOOST_CHECK_THROW(
      opt2.validReplacement(SGD({{"defaultLearningRate", {2, true}},
                                 {"defaultWeightDecay", {1, true}},
                                 {"lossScaling", {1, true}}})),
      popart::error);

  BOOST_CHECK_THROW(
      opt2.validReplacement(SGD({{"defaultLearningRate", {1, true}},
                                 {"defaultWeightDecay", {2, true}},
                                 {"lossScaling", {1, true}}})),
      popart::error);

  BOOST_CHECK_THROW(
      opt2.validReplacement(SGD({{"defaultLearningRate", {1, true}},
                                 {"defaultWeightDecay", {1, true}},
                                 {"lossScaling", {2, true}}})),
      popart::error);

  //  Tensor foo has lr 2 and wd 2
  auto opt3 = SGD({{"defaultLearningRate", {1, true}},
                   {"defaultWeightDecay", {1, true}},
                   {"lossScaling", {1, true}}});
  opt3.insertSpecific(
      "foo", {{"learningRate", {2, true}}, {"weightDecay", {2, true}}});

  //  Tensor foo has lr 3 and wd 3
  auto opt4 = SGD({{"defaultLearningRate", {1, true}},
                   {"defaultWeightDecay", {1, true}},
                   {"lossScaling", {1, true}}});
  opt4.insertSpecific(
      "foo", {{"learningRate", {3, true}}, {"weightDecay", {3, true}}});

  // Cannot change specific value if it is const
  BOOST_CHECK_THROW(opt3.validReplacement(opt4), popart::error);
  BOOST_CHECK_THROW(opt4.validReplacement(opt3), popart::error);
}
