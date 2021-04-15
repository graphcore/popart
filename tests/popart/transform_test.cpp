// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE TransformTest

#include <boost/test/unit_test.hpp>

#define private public
#include <popart/ir.hpp>
#include <popart/transforms/transform.hpp>
#undef private

using namespace popart;

class TestTransform : public Transform {
public:
  static std::size_t id() { return typeid(TestTransform).hash_code(); }

  TestTransform() : Transform() {}
  virtual ~TestTransform() override {}

  virtual bool apply(Graph &) const final {
    executedCount++;
    return true;
  }

  virtual std::size_t getId() const final { return id(); }

  virtual std::string getName() const final { return "TestTransform"; }

  static int executedCount;
};

int TestTransform::executedCount = 0;

namespace {
bool init = Transform::registerTransform(new TestTransform);
}

BOOST_AUTO_TEST_CASE(TransformDefault) {

  Ir ir;

  int initialCount = TestTransform::executedCount;

  ir.applyTransformIfEnabled(TestTransform::id(), ir.getMainGraph());

  BOOST_CHECK_EQUAL(TestTransform::executedCount, initialCount + 1);
}

BOOST_AUTO_TEST_CASE(TransformEnabled) {
  Ir ir;

  int initialCount = TestTransform::executedCount;

  ir.enableTransform(TestTransform::id(), true);
  ir.applyTransformIfEnabled(TestTransform::id(), ir.getMainGraph());

  BOOST_CHECK_EQUAL(TestTransform::executedCount, initialCount + 1);
}

BOOST_AUTO_TEST_CASE(TransformDisabled) {
  Ir ir;

  int initialCount = TestTransform::executedCount;

  ir.enableTransform(TestTransform::id(), false);
  ir.applyTransformIfEnabled(TestTransform::id(), ir.getMainGraph());

  BOOST_CHECK_EQUAL(TestTransform::executedCount, initialCount);
}

BOOST_AUTO_TEST_CASE(TransformDefaultGenericConditional) {

  Ir ir;

  int initialCount = TestTransform::executedCount;

  ir.applyTransformIfEnabled<TestTransform>(std::ref(ir.getMainGraph()));

  BOOST_CHECK_EQUAL(TestTransform::executedCount, initialCount + 1);
}

BOOST_AUTO_TEST_CASE(TransformEnabledGenericConditional) {
  Ir ir;

  int initialCount = TestTransform::executedCount;

  ir.enableTransform(TestTransform::id(), true);
  ir.applyTransformIfEnabled<TestTransform>(std::ref(ir.getMainGraph()));

  BOOST_CHECK_EQUAL(TestTransform::executedCount, initialCount + 1);
}

BOOST_AUTO_TEST_CASE(TransformDisabledGenericConditional) {
  Ir ir;

  int initialCount = TestTransform::executedCount;

  ir.enableTransform(TestTransform::id(), false);
  ir.applyTransformIfEnabled<TestTransform>(std::ref(ir.getMainGraph()));

  BOOST_CHECK_EQUAL(TestTransform::executedCount, initialCount);
}

BOOST_AUTO_TEST_CASE(TransformDefaultGenericUnconditional) {

  Ir ir;

  int initialCount = TestTransform::executedCount;

  ir.applyTransform<TestTransform>(std::ref(ir.getMainGraph()));

  BOOST_CHECK_EQUAL(TestTransform::executedCount, initialCount + 1);
}

BOOST_AUTO_TEST_CASE(TransformEnabledGenericUnconditional) {
  Ir ir;

  int initialCount = TestTransform::executedCount;

  ir.enableTransform(TestTransform::id(), true);
  ir.applyTransform<TestTransform>(std::ref(ir.getMainGraph()));

  BOOST_CHECK_EQUAL(TestTransform::executedCount, initialCount + 1);
}

BOOST_AUTO_TEST_CASE(TransformDisabledGenericUnconditional) {
  Ir ir;

  int initialCount = TestTransform::executedCount;

  ir.enableTransform(TestTransform::id(), false);
  ir.applyTransform<TestTransform>(std::ref(ir.getMainGraph()));

  // Application of transform is unconditional.
  BOOST_CHECK_EQUAL(TestTransform::executedCount, initialCount + 1);
}
