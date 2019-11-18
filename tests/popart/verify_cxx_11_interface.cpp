#define BOOST_TEST_MODULE VerifyCxx11Interface

#include <boost/test/unit_test.hpp>

#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/devicemanager.hpp>
#include <popart/error.hpp>
#include <popart/filereader.hpp>
#include <popart/graph.hpp>
#include <popart/graphtransformer.hpp>
#include <popart/half.hpp>
#include <popart/inputshapeinfo.hpp>
#include <popart/ir.hpp>
#include <popart/logging.hpp>
#include <popart/names.hpp>
#include <popart/ndarraywrapper.hpp>
#include <popart/onnxutil.hpp>
#include <popart/op.hpp>
#include <popart/op/add.hpp>
#include <popart/op/if.hpp>
#include <popart/op/ipucopy.hpp>
#include <popart/op/l1.hpp>
#include <popart/op/matmul.hpp>
#include <popart/op/nll.hpp>
#include <popart/op/restore.hpp>
#include <popart/op/stash.hpp>
#include <popart/op/varupdate.hpp>
#include <popart/opmanager.hpp>
#include <popart/optimizer.hpp>
#include <popart/patterns/pattern.hpp>
#include <popart/patterns/patterns.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/matmulx.hpp>
#include <popart/popx/opx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/region.hpp>
#include <popart/session.hpp>
#include <popart/sessionoptions.hpp>
#include <popart/subgraph/algo0.hpp>
#include <popart/subgraph/algo1.hpp>
#include <popart/subgraph/isomorphic.hpp>
#include <popart/subgraph/match.hpp>
#include <popart/subgraph/outliner.hpp>
#include <popart/tensor.hpp>
#include <popart/tensordata.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/tensornames.hpp>
#include <popart/tensors.hpp>
#include <popart/topocons.hpp>
#include <popart/transforms/prune.hpp>
#include <popart/transforms/transform.hpp>

BOOST_AUTO_TEST_CASE(Basic0) {
  // To check this test is actually being built for c++11, uncomment the below
  // and try to build. It should fail to build for c++11.
  //
  // auto foo = [](auto lhs, auto rhs) {
  //   return lhs / rhs;
  // };
  // BOOST_CHECK_EQUAL(foo(3.0f, 4.0f), 0.75f);
  // BOOST_CHECK_EQUAL(foo(3, 4), 0);
}
