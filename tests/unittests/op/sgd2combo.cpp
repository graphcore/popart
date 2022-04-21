// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE Test_Op_SGD2Combo
#include <boost/test/unit_test.hpp>

#include <popart/error.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/sgd2combo.hpp>
#include <popart/optimizer.hpp>

using namespace popart;

namespace {
using TestFunc = std::function<void(Graph &)>;

struct SGD2CtorValidationTestFixture {
  Ir ir;

  void requireThrows(TestFunc testF) {
    BOOST_REQUIRE_THROW(testF(ir.getMainGraph()), error);
  }

  void requireNoThrow(TestFunc testF) {
    BOOST_REQUIRE_NO_THROW(testF(ir.getMainGraph()));
  }
};
} // namespace

BOOST_AUTO_TEST_CASE(TestCtorValidation) {
  SGD2CtorValidationTestFixture tf;

  // Rejects AcclReduce.
  tf.requireThrows([](Graph &g) {
    SGD2ComboOp{{},
                {},
                {},
                {},
                true,
                OptimizerReductionType::AcclReduce,
                DataType::FLOAT,
                DataType::FLOAT,
                Op::Settings{g, "SGD2Combo"}};
  });

  // Rejects no grad acc + AccumReduce.
  tf.requireThrows([](Graph &g) {
    SGD2ComboOp{{},
                {},
                {},
                {},
                false, // grad acc off.
                OptimizerReductionType::AccumReduce,
                DataType::FLOAT,
                DataType::FLOAT,
                Op::Settings{g, "SGD2Combo"}};
  });

  // Accepts grad acc + AccumReduce.
  tf.requireNoThrow([](Graph &g) {
    SGD2ComboOp{{},
                {},
                {},
                {},
                true, // grad acc on.
                OptimizerReductionType::AccumReduce,
                DataType::FLOAT,
                DataType::FLOAT,
                Op::Settings{g, "SGD2Combo"}};
  });

  // Accepts None reduction.
  tf.requireNoThrow([](Graph &g) {
    SGD2ComboOp{{},
                {},
                {},
                {},
                false,
                OptimizerReductionType::None,
                DataType::FLOAT,
                DataType::FLOAT,
                Op::Settings{g, "SGD2Combo"}};
  });

  // Accepts only FLOAT and FLOAT16 DataTypes for accum and accl1.
  tf.requireThrows([](Graph &g) {
    SGD2ComboOp{{},
                {},
                {},
                {},
                true,
                OptimizerReductionType::AccumReduce,
                DataType::UINT16,
                DataType::FLOAT,
                Op::Settings{g, "SGD2Combo"}};
  });
  tf.requireThrows([](Graph &g) {
    SGD2ComboOp{{},
                {},
                {},
                {},
                true,
                OptimizerReductionType::AccumReduce,
                DataType::FLOAT,
                DataType::STRING,
                Op::Settings{g, "SGD2Combo"}};
  });
  tf.requireNoThrow([](Graph &g) {
    SGD2ComboOp{{},
                {},
                {},
                {},
                false,
                OptimizerReductionType::None,
                DataType::FLOAT16,
                DataType::FLOAT,
                Op::Settings{g, "SGD2Combo"}};
  });
  tf.requireNoThrow([](Graph &g) {
    SGD2ComboOp{{},
                {},
                {},
                {},
                true,
                OptimizerReductionType::GradReduce,
                DataType::FLOAT,
                DataType::FLOAT16,
                Op::Settings{g, "SGD2Combo"}};
  });
}
