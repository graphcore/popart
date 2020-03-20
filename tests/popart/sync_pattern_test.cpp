// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE SyncPatternTest

#include <boost/test/unit_test.hpp>

#include <popart/devicemanager.hpp>
#include <popart/popx/devicexmanager.hpp>
#include <popart/testdevice.hpp>

using namespace popart;
using namespace popart::popx;

// TODO(T16464): Expected to pass in future Poplar version
BOOST_AUTO_TEST_CASE(SyncPatternTest_0) {

  auto deviceInfo0 =
      createTestDevice(TEST_TARGET, 2, 1216, SyncPattern::SinglePipeline);

  if (deviceInfo0) {
    DevicexInfo &di0 = dynamic_cast<DevicexInfo &>(*deviceInfo0);
    poplar::Graph graph(
        di0.getDevice().getTarget(), 0, poplar::replication_factor(1));
    auto exe = poplar::compileGraph(graph, {poplar::program::Sequence()});
    poplar::Engine engine(std::move(exe));
    engine.load(di0.getDevice());
    engine.run();
  }
}

// TODO(T16464): Expected to fail in future Poplar version
BOOST_AUTO_TEST_CASE(SyncPatternTest_1) {

  auto deviceInfo0 =
      createTestDevice(TEST_TARGET, 2, 1216, SyncPattern::SinglePipeline);

  auto deviceInfo1 = createTestDevice(TEST_TARGET, 2, 1216);

  if (deviceInfo0 && deviceInfo1) {
    DevicexInfo &di0 = dynamic_cast<DevicexInfo &>(*deviceInfo0);
    DevicexInfo &di1 = dynamic_cast<DevicexInfo &>(*deviceInfo1);
    poplar::Graph graph(
        di0.getDevice().getTarget(), 0, poplar::replication_factor(1));
    auto exe = poplar::compileGraph(graph, {poplar::program::Sequence()});
    poplar::Engine engine(std::move(exe));
    engine.load(di1.getDevice());
    engine.run();
  }
}

// TODO(T16464): Expected to pass in future Poplar version
BOOST_AUTO_TEST_CASE(SyncPatternTest_2, *boost::unit_test::disabled()) {

  auto deviceInfo0 =
      createTestDevice(TEST_TARGET, 2, 1216, SyncPattern::PingPong);

  if (deviceInfo0) {
    DevicexInfo &di0 = dynamic_cast<DevicexInfo &>(*deviceInfo0);
    poplar::Graph graph(
        di0.getDevice().getTarget(), 0, poplar::replication_factor(1));
    auto exe = poplar::compileGraph(graph, {poplar::program::Sequence()});
    poplar::Engine engine(std::move(exe));
    engine.load(di0.getDevice());
    engine.run();
  }
}

// TODO(T16464): Expected to fail in future Poplar version
BOOST_AUTO_TEST_CASE(SyncPatternTest_3, *boost::unit_test::disabled()) {

  auto deviceInfo0 =
      createTestDevice(TEST_TARGET, 2, 1216, SyncPattern::PingPong);
  auto deviceInfo1 = createTestDevice(TEST_TARGET, 2, 1216);

  if (deviceInfo0 && deviceInfo1) {
    DevicexInfo &di0 = dynamic_cast<DevicexInfo &>(*deviceInfo0);
    DevicexInfo &di1 = dynamic_cast<DevicexInfo &>(*deviceInfo1);
    poplar::Graph graph(
        di0.getDevice().getTarget(), 0, poplar::replication_factor(1));
    auto exe = poplar::compileGraph(graph, {poplar::program::Sequence()});
    poplar::Engine engine(std::move(exe));
    engine.load(di1.getDevice());
    engine.run();
  }
}

// TODO(T16464): Expected to fail in future Poplar version
BOOST_AUTO_TEST_CASE(SyncPatternTest_4, *boost::unit_test::disabled()) {

  auto deviceInfo0 =
      createTestDevice(TEST_TARGET, 2, 1216, SyncPattern::PingPong);

  auto deviceInfo1 =
      createTestDevice(TEST_TARGET, 2, 1216, SyncPattern::SinglePipeline);
  if (deviceInfo0 && deviceInfo1) {
    DevicexInfo &di0 = dynamic_cast<DevicexInfo &>(*deviceInfo0);
    DevicexInfo &di1 = dynamic_cast<DevicexInfo &>(*deviceInfo1);
    poplar::Graph graph(
        di0.getDevice().getTarget(), 0, poplar::replication_factor(1));
    auto exe = poplar::compileGraph(graph, {poplar::program::Sequence()});
    poplar::Engine engine(std::move(exe));
    engine.load(di1.getDevice());
    engine.run();
  }
}
