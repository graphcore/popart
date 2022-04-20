// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE TransformTest

#include <boost/test/unit_test.hpp>
#include <popart/transforms/transform.hpp>

#ifdef __clang__
#pragma clang diagnostic ignored "-Wkeyword-macro"
#endif
#define private public
#include <cstddef>
#include <string>
#include <typeinfo>
#include <popart/ir.hpp>

namespace popart {
class Graph;
} // namespace popart

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

  ir.applyTransform(TestTransform::id(), ir.getMainGraph());

  BOOST_CHECK_EQUAL(TestTransform::executedCount, initialCount + 1);
}

BOOST_AUTO_TEST_CASE(TransformEnabled) {
  Ir ir;

  int initialCount = TestTransform::executedCount;

  ir.enableTransform(TestTransform::id(), true);
  ir.applyTransform(TestTransform::id(), ir.getMainGraph());

  BOOST_CHECK_EQUAL(TestTransform::executedCount, initialCount + 1);
}

BOOST_AUTO_TEST_CASE(TransformDisabled) {
  Ir ir;

  int initialCount = TestTransform::executedCount;

  ir.enableTransform(TestTransform::id(), false);
  ir.applyTransform(TestTransform::id(), ir.getMainGraph());

  BOOST_CHECK_EQUAL(TestTransform::executedCount, initialCount);
}
