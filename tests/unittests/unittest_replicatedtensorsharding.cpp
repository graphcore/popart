// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE TensorIdTest

#include <testutil/irquery/irquery.hpp>
#include <testutil/test_graphs/graph_test_models.hpp>

#include <boost/test/unit_test.hpp>

#include <popart/op/copyvarupdate.hpp>
#include <popart/op/exchange/multiexchange.hpp>
#include <popart/pointercomparators.hpp>
#include <popart/replicatedtensorsharding.hpp>

using namespace popart;
using namespace popart::irquery;

BOOST_AUTO_TEST_CASE(checkRTSDomainTest0) {

  SessionOptions options;

  options.replicatedGraphCount = 4;

  RemoteRTSTestModel model(options);

  popart::ReplicatedTensorShardingTracer tracer(model.getIr());

  // Trace all domains separately from the start tensor
  {
    std::set<Tensor *, PTensorCmp> startTensors{model.domainTensors.at(0)};
    tracer.trace(startTensors);
  }

  {
    std::set<Tensor *, PTensorCmp> startTensors{model.domainTensors.at(1)};
    tracer.trace(startTensors);
  }

  {
    std::set<Tensor *, PTensorCmp> startTensors{model.domainTensors.at(2)};
    tracer.trace(startTensors);
  }

  {
    std::set<Tensor *, PTensorCmp> startTensors{model.domainTensors.at(3)};
    tracer.trace(startTensors);
  }

  // Check presence of MultiExchangeOp
  // This is important so that we can ensure the groups are correct and
  // MultiExchangeOp does not spill separate RTS domains into each other
  IrTestWrapper tw_ir{model.getIr()};
  auto tw_mainGraph =
      tw_ir.hasGraph(model.getIr().getMainGraph().id, Require::MustBeTrue);

  auto tw_multiExchangeOp = tw_mainGraph->ops().hasOp<MultiExchangeOp>(
      [&](auto &tw_op) -> bool {
        Op *op = tw_op.unwrap();

        // InitOp outputs and offsets
        if (op->input->n() != 8) {
          return false;
        }

        // Loaded tensors
        if (op->output->n() != 4) {
          return false;
        }

        bool nextExpectedOpCorrect = true;

        for (auto &output : op->output->tensorMap()) {
          logging::trace(
              "[checkRTSDomainTest0] Next Op: {}",
              output.second->consumers.getOps().front()->debugName());

          nextExpectedOpCorrect &= output.second->consumers.getOps()
                                       .front()
                                       ->isConvertibleTo<CopyVarUpdateOp>();
        }

        // Check that ops and tensors have been grouped correctly

        // First group traced
        BOOST_CHECK_EQUAL(tracer.getGroup(op->output->tensorMap().at(0)->id).id,
                          0);
        // Second group traced
        BOOST_CHECK_EQUAL(tracer.getGroup(op->output->tensorMap().at(1)->id).id,
                          1);
        // Third tensor is the first group again due to a shared remote buffer
        // ID
        BOOST_CHECK_EQUAL(tracer.getGroup(op->output->tensorMap().at(2)->id).id,
                          0);
        // Fourth group traced
        BOOST_CHECK_EQUAL(tracer.getGroup(op->output->tensorMap().at(3)->id).id,
                          2);

        return nextExpectedOpCorrect;
      },
      Require::MustBeTrue);

  // Check all groups are correct

  // First group traced
  BOOST_CHECK_EQUAL(tracer.getGroup(model.domainTensors.at(0)->id).id, 0);
  // Second group traced
  BOOST_CHECK_EQUAL(tracer.getGroup(model.domainTensors.at(1)->id).id, 1);
  // Third tensor is the first group again due to a shared remote buffer ID
  BOOST_CHECK_EQUAL(tracer.getGroup(model.domainTensors.at(2)->id).id, 0);
  // Fourth group traced
  BOOST_CHECK_EQUAL(tracer.getGroup(model.domainTensors.at(3)->id).id, 2);
}
