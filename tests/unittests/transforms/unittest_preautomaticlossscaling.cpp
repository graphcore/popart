// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE PreAutomaticLossScaleTest
#include <algorithm>
#include <iterator>
#include <map>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <boost/algorithm/string/predicate.hpp>
#include <boost/test/unit_test.hpp>

#include <popart/error.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/names.hpp>
#include <popart/op/autolossscaleproxy.hpp>
#include <popart/op/mul.hpp>
#include <popart/tensorid.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/transforms/preautomaticlossscaling.hpp>
#include <popart/util.hpp>

namespace popart {

void addMulOp(TensorId, TensorId, TensorId, Graph &, int);
template <typename Ex>
std::function<bool(const Ex &)> checkErrorMsgHasPrefixFn(const std::string &);

/**
 * Test whether the PreAutomaticLossScale transform inserts AutoLossScaleProxyOp
 * in the right places.
 *
 * To do this, an initial graph is built and then transformed using
 * PreAutomaticLossScale, given a list of tensors. The connections of each
 * tensor in the transformed graph are then inspected.
 *
 * Initial graph:
 *
 *   A -- M0 -- D -- M2 -------- F
 *   B -'            |
 *                   |
 *   C == M1 ------- E == M3 -- G
 *
 * The transform should add an AutoLossScaleProxyOp (annotated as [P] below)
 * after each of the given tensor names. Each [P] acts as an identity op. The
 * transformed graph, given all tensor names in the graph, would look as
 * follows:
 *
 * A - [P] - A* -- M0 -- D - [P] - D* -- M2 -- F - [P] - F*
 * B - [P] - B* -'                       |
 *                                       |
 * C - [P] - C* == M1 ------------------ E - [P] - E* == M3 -- G - [P] - G*
 */
BOOST_AUTO_TEST_CASE(TestAutoLossScaleProxyOpInserted) {
  auto test = [](std::vector<TensorId> toTrackTensors) {
    Ir ir;
    Graph &g = ir.getMainGraph();

    // Ensure toTrackTensors are sorted.
    std::sort(toTrackTensors.begin(), toTrackTensors.end());
    // All tensor IDs in the graph before the transform.
    std::vector<std::string> allTensors = {"A", "B", "C", "D", "E", "F", "G"};
    // The non-tracked tensor IDs.
    std::vector<std::string> noTrackTensors;
    std::set_difference(allTensors.begin(),
                        allTensors.end(),
                        toTrackTensors.begin(),
                        toTrackTensors.end(),
                        std::inserter(noTrackTensors, noTrackTensors.begin()));

    auto &alsSettings = ir.getSessionOptions().automaticLossScalingSettings;
    alsSettings.toTrackTensors = toTrackTensors;

    // Create the input tensors to the graph.
    TensorInfo tInfo = {DataType::FLOAT, Shape{1}};
    g.getTensors().addVarInit("A", tInfo, "A");
    g.getTensors().addVarInit("B", tInfo, "B");
    g.getTensors().addVarInit("C", tInfo, "C");

    // Build the graph.
    addMulOp("A", "B", "D", g, 0);
    addMulOp("C", "C", "E", g, 1);
    addMulOp("D", "E", "F", g, 2);
    addMulOp("E", "E", "G", g, 3);

    // Store consumers and producers of each tensor before applying the
    // transform.
    std::map<TensorId, std::string> oldProducers                     = {};
    std::map<TensorId, std::unordered_set<std::string>> oldConsumers = {};

    for (auto id : g.getTensors().getAllTensorIds()) {
      auto tensor = g.getTensors().get(id);

      std::string producer = "";
      if (tensor->hasProducer()) {
        producer = tensor->getProducer()->getName();
      }
      oldProducers[id] = producer;

      std::unordered_set<std::string> consumers = {};
      for (auto op : tensor->consumers.getOps()) {
        auto opName  = op->getName();
        auto indices = op->input->indicesMap().at(tensor);
        for (auto i : indices) {
          consumers.insert(opName + "/" + std::to_string(i));
        }
      }
      oldConsumers[id] = consumers;
    }

    // Apply the transform.
    PreAutomaticLossScale transform;
    bool applyResult = transform.apply(g);
    BOOST_TEST(applyResult == true);

    // Examine the transformed parts of the graph.
    for (auto id : alsSettings.toTrackTensors.value()) {
      auto tensor = ir.getTensor(id);

      // Verify that the producer hasn't changed.
      if (tensor->hasProducer()) {
        BOOST_TEST(tensor->getProducer()->getName() == oldProducers[id]);
      } else {
        BOOST_TEST("" == oldProducers[id]);
      }

      // Verify that there's only one consumer, and that's
      // AutoLossScaleProxyOp.
      BOOST_TEST(tensor->consumers.getOps().size() == 1);
      BOOST_TEST(tensor->consumers.getOps()[0]->opid ==
                 Onnx::CustomOperators::AutoLossScaleProxy);

      // Verify that the tensor, produced by AutoLossScaleProxyOp, is
      // connected to the same consumers as the original tensor.
      auto proxyTensor = tensor->consumers.getOps()[0]->output->tensor(0);

      std::unordered_set<std::string> consumers = {};
      for (auto op : proxyTensor->consumers.getOps()) {
        auto opName  = op->getName();
        auto indices = op->input->indicesMap().at(proxyTensor);
        for (auto i : indices) {
          consumers.insert(opName + "/" + std::to_string(i));
        }
      }
      BOOST_TEST(oldConsumers[id] == consumers);
    }

    // Examine the non-transformed parts of the graph.
    for (auto id : noTrackTensors) {
      // Skip, if tensor is tracked.
      if (std::find(toTrackTensors.begin(), toTrackTensors.end(), id) !=
          toTrackTensors.end()) {
        continue;
      }

      auto tensor = g.getTensors().get(id);

      // Verify that producer is unchanged.
      std::string producer = "";
      if (tensor->hasProducer()) {
        producer = tensor->getProducer()->getName();
      }
      BOOST_TEST(producer == oldProducers[id]);

      // Verify that consumers are unchanged.
      std::unordered_set<std::string> consumers = {};
      for (auto op : tensor->consumers.getOps()) {
        auto opName  = op->getName();
        auto indices = op->input->indicesMap().at(tensor);
        for (auto i : indices) {
          consumers.insert(opName + "/" + std::to_string(i));
        }
      }
      BOOST_TEST(consumers == oldConsumers[id]);
    }
  };

  test({"A", "B", "C", "D", "E", "F", "G"});
  test({"A", "B", "C", "E", "F", "G"});
  test({"A", "B", "C", "D", "E", "F"});
  test({"B", "C", "D", "E", "F", "G"});
  test({"A", "D", "E", "G"});
  test({"A", "B", "C"});
  test({"D", "E", "F"});
  for (TensorId id : {"A", "B", "C", "D", "E", "F", "G"})
    test({id});
}

/**
 * Verify that the transform leaves the graph unchanged if the user doesn't set
 * automaticLossScalingSettings.toTrackTensors.
 *
 * Here unchanged means that the IR is in the exact same state as it was before
 * attempting to apply the transform. However, this test only checks whether the
 * hash of the serialized IR is unchanged. This means that the state of TopoCons
 * and some Op members will not be included in the comparison. However, the
 * general structure of the graph - all Ops with their inputs, outputs, and
 * attributes are included.
 */
BOOST_AUTO_TEST_CASE(TestToTrackTensorsUnset) {
  Ir ir;
  Graph &g = ir.getMainGraph();

  ir.computeHash();
  auto hashBefore = ir.getHash();

  PreAutomaticLossScale transform;
  bool applyResult = transform.apply(g);
  BOOST_TEST(applyResult == false);

  ir.computeHash();
  auto hashAfter = ir.getHash();

  BOOST_TEST(hashBefore == hashAfter);
}

/**
 * Verify that an error is thrown when the toTrackTensors list is empty.
 */
BOOST_AUTO_TEST_CASE(TestNoToTrackTensors) {
  Ir ir;
  Graph &g = ir.getMainGraph();

  auto &alsSettings = ir.getSessionOptions().automaticLossScalingSettings;
  alsSettings.toTrackTensors = std::vector<TensorId>{};

  // Apply the transform and verify that an error is thrown.
  PreAutomaticLossScale transform;
  const auto checkErrorFn = checkErrorMsgHasPrefixFn<error>(
      std::string("[PreAutomaticLossScale] An empty list was set as the value "
                  "of 'toTrackTensors'."));
  BOOST_REQUIRE_EXCEPTION(transform.apply(g), error, checkErrorFn);
}

/**
 * Verify that an error is thrown when the toTrackTensors list contains tensors
 * that are not in the graph.
 */
BOOST_AUTO_TEST_CASE(TestToTrackTensorsNotInGraph) {
  auto test = [](std::vector<TensorId> toTrackTensors) {
    Ir ir;
    Graph &g = ir.getMainGraph();

    auto &alsSettings = ir.getSessionOptions().automaticLossScalingSettings;
    alsSettings.toTrackTensors = toTrackTensors;

    // Create the input tensors to the graph.
    TensorInfo tInfo = {DataType::FLOAT, Shape{1}};
    g.getTensors().addVarInit("A", tInfo, "A");
    g.getTensors().addVarInit("B", tInfo, "B");

    // Build the graph.
    addMulOp("A", "B", "C", g, 0);

    // Apply the transform and verify that an error is thrown.
    PreAutomaticLossScale transform;
    const auto checkErrorFn = checkErrorMsgHasPrefixFn<error>(
        std::string("[PreAutomaticLossScale] Some of the tensors in the "
                    "'automaticLossScalingSettings.toTrackTensors' list do not "
                    "exist in the model."));
    BOOST_REQUIRE_EXCEPTION(transform.apply(g), error, checkErrorFn);
  };

  test({"D", "E", "F"});
  test({"A", "E"});
  test({"E"});
}

/**
 * Adds a MulOp to the graph.
 *
 * \param arg0 The first input argument to the MulOp.
 * \param arg1 The second input argument to the MulOp.
 * \param out The output of the MulOp.
 * \param g The graph this will be added in.
 * \param id The # (ID) of the MulOp - this will be added to its name as M#.
 */
void addMulOp(TensorId arg0, TensorId arg1, TensorId out, Graph &g, int id) {
  g.createConnectedOp<MulOp>(
      {{MulOp::getArg0InIndex(), arg0}, {MulOp::getArg1InIndex(), arg1}},
      {{MulOp::getOutIndex(), out}},
      Onnx::Operators::Mul_7,
      Op::Settings(g, "M" + std::to_string(id)));
}

template <typename Ex>
std::function<bool(const Ex &)>
checkErrorMsgHasPrefixFn(const std::string &prefix) {
  return [=](const Ex &ex) -> bool {
    return boost::algorithm::starts_with(ex.what(), prefix);
  };
}
} // namespace popart
