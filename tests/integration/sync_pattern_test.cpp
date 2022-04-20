// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE SyncPatternTest

#include <boost/test/unit_test.hpp>
#include <memory>
#include <snap/Graph.hpp>
#include <utility>
#include <poplar/Device.hpp>
#include <poplar/Engine.hpp>
#include <poplar/Program.hpp>
#include <popart/devicemanager.hpp>
#include <popart/popx/devicexmanager.hpp>
#include <popart/testdevice.hpp>

namespace poplar {
struct poplar_error;
} // namespace poplar

using namespace popart;
using namespace popart::popx;

BOOST_AUTO_TEST_CASE(SyncPatternTest_0) {

  auto deviceInfo0 =
      createTestDevice(TEST_TARGET, 2, 0, SyncPattern::SinglePipeline);

  if (deviceInfo0) {
    DevicexInfo &di0 = dynamic_cast<DevicexInfo &>(*deviceInfo0);
    snap::Graph graph(di0.getDevice().getTarget());
    auto exe = poplar::compileGraph(graph.getPoplarGraph(),
                                    {poplar::program::Sequence()});
    poplar::Engine engine(std::move(exe));
    engine.load(di0.getDevice());
    engine.run();
  }
}

BOOST_AUTO_TEST_CASE(SyncPatternTest_1) {

  auto deviceInfo0 =
      createTestDevice(TEST_TARGET, 2, 0, SyncPattern::SinglePipeline);

  auto deviceInfo1 = createTestDevice(TEST_TARGET, 2);

  if (deviceInfo0 && deviceInfo1) {
    DevicexInfo &di0 = dynamic_cast<DevicexInfo &>(*deviceInfo0);
    DevicexInfo &di1 = dynamic_cast<DevicexInfo &>(*deviceInfo1);
    snap::Graph graph(di0.getDevice().getTarget());
    auto exe = poplar::compileGraph(graph.getPoplarGraph(),
                                    {poplar::program::Sequence()});
    poplar::Engine engine(std::move(exe));
    // Will throw a  'Attempt to load graph compiled with target options
    // {{IpuLinkConfiguration: 0}, {SyncConfiguration: 1}} onto a device with
    // {{IpuLinkConfiguration: 0}, {SyncConfiguration: 0}}' error
    // Similarly for tests below.
    BOOST_CHECK_THROW(engine.load(di1.getDevice()), poplar::poplar_error);
    di1.detach(); // avoid ipu attachment conflicts

    engine.load(di0.getDevice());
    engine.run();
  }
}

BOOST_AUTO_TEST_CASE(SyncPatternTest_2) {

  auto deviceInfo0 =
      createTestDevice(TEST_TARGET, 2, 0, SyncPattern::ReplicaAndLadder);

  if (deviceInfo0) {
    DevicexInfo &di0 = dynamic_cast<DevicexInfo &>(*deviceInfo0);
    snap::Graph graph(di0.getDevice().getTarget());
    auto exe = poplar::compileGraph(graph.getPoplarGraph(),
                                    {poplar::program::Sequence()});
    poplar::Engine engine(std::move(exe));
    engine.load(di0.getDevice());
    engine.run();
  }
}

BOOST_AUTO_TEST_CASE(SyncPatternTest_3) {

  auto deviceInfo0 =
      createTestDevice(TEST_TARGET, 2, 0, SyncPattern::ReplicaAndLadder);
  auto deviceInfo1 = createTestDevice(TEST_TARGET, 2);

  if (deviceInfo0 && deviceInfo1) {
    DevicexInfo &di0 = dynamic_cast<DevicexInfo &>(*deviceInfo0);
    DevicexInfo &di1 = dynamic_cast<DevicexInfo &>(*deviceInfo1);
    snap::Graph graph(di0.getDevice().getTarget());
    auto exe = poplar::compileGraph(graph.getPoplarGraph(),
                                    {poplar::program::Sequence()});
    poplar::Engine engine(std::move(exe));
    BOOST_CHECK_THROW(engine.load(di1.getDevice()), poplar::poplar_error);
    di1.detach(); // avoid ipu attachment conflicts

    engine.load(di0.getDevice());
    engine.run();
  }
}

BOOST_AUTO_TEST_CASE(SyncPatternTest_4) {

  auto deviceInfo0 =
      createTestDevice(TEST_TARGET, 2, 0, SyncPattern::ReplicaAndLadder);

  auto deviceInfo1 =
      createTestDevice(TEST_TARGET, 2, 0, SyncPattern::SinglePipeline);
  if (deviceInfo0 && deviceInfo1) {
    DevicexInfo &di0 = dynamic_cast<DevicexInfo &>(*deviceInfo0);
    DevicexInfo &di1 = dynamic_cast<DevicexInfo &>(*deviceInfo1);
    snap::Graph graph(di0.getDevice().getTarget());
    auto exe = poplar::compileGraph(graph.getPoplarGraph(),
                                    {poplar::program::Sequence()});
    poplar::Engine engine(std::move(exe));
    BOOST_CHECK_THROW(engine.load(di1.getDevice()), poplar::poplar_error);
    di1.detach(); // avoid ipu attachment conflicts

    engine.load(di0.getDevice());
    engine.run();
  }
}
