#define BOOST_TEST_MODULE IsNormTest

#include <boost/test/unit_test.hpp>
#include <onnx/onnx_pb.h>
#include <popart/builder.hpp>
#include <popart/ir.hpp>
#include <popart/names.hpp>
#include <popart/opmanager.hpp>

BOOST_AUTO_TEST_CASE(IsNormTest) {
  using namespace popart;
  popart::Ir ir;

  // Is 'add' a norm? - No
  std::unique_ptr<Op> add =
      OpManager::createOp(Onnx::Operators::Add_7, ir.getMainGraph());
  BOOST_CHECK(!add.get()->isNorm());

  // Is 'batchnorm' a norm? - Yes
  std::unique_ptr<Op> bn = OpManager::createOp(
      Onnx::Operators::BatchNormalization_9, ir.getMainGraph());
  BOOST_CHECK(bn.get()->isNorm());

  // Is 'groupnorm' a norm? - Yes
  Node node;
  node.add_attribute();
  node.mutable_attribute(0)->set_name("num_groups");
  node.mutable_attribute(0)->set_i(1);
  NodeAttributes nodeAttr = node.attribute();
  Attributes attr(nodeAttr);
  std::unique_ptr<Op> gn = OpManager::createOp(
      Onnx::CustomOperators::GroupNormalization_1, ir.getMainGraph(), "", attr);
  BOOST_CHECK(gn.get()->isNorm());

  // Is 'instancenorm' a norm? - Yes
  std::unique_ptr<Op> in = OpManager::createOp(
      Onnx::Operators::InstanceNormalization_6, ir.getMainGraph());
  BOOST_CHECK(in.get()->isNorm());
}
