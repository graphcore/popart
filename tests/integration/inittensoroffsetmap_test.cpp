// Copyright (c) 2023 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE init_tensor_offset_map

#include <algorithm>
#include <any>
#include <boost/test/unit_test.hpp>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <poplar/Graph.hpp>
#include <poplar/Interval.hpp>

#include "popart/builder.gen.hpp"
#include "popart/dataflow.hpp"
#include "popart/debugcontext.hpp"
#include "popart/inputshapeinfo.hpp"
#include "popart/logging.hpp"
#include "popart/names.hpp"
#include "popart/patterns/patterns.hpp"
#include "popart/sessionoptions.hpp"
#include "popart/tensorinfo.hpp"
#include "popart/util.hpp"
#include "popart/voiddata.hpp"

// This trick is required to access the Devicex's poplar::Tensors.

#ifdef __clang__
#pragma clang diagnostic ignored "-Wkeyword-macro"
#endif
#define protected public
#define private public

#include <testdevice.hpp>
#include <popart/builder.hpp>
#include <popart/devicemanager.hpp>
#include <popart/error.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/irlowering.hpp>
#include <popart/session.hpp>
#include <popart/sgd.hpp>

#include "popart/popx/poptensors.hpp"

#undef private
#undef protected

BOOST_AUTO_TEST_CASE(InitTensorOffsetMap) {
  // In this test, the input tensors are the exact size of a packet bytes for
  // one tile, Therefore, when createHostTransferableTensorWithOffset = true,
  // the accumulated tensor bytes is passed to createHostTransferableTensor()
  // as offset, and it mapping those tensors across tiles rather than mapping
  // them all to tile0.

  using namespace popart;

  auto builder = Builder::create();
  auto aiOnnx  = builder->aiOnnxOpset9();

  // one packet per tile = 1024 bytes = 256 * FLOAT
  std::vector<int64_t> inputShape{1, 256};
  TensorInfo inputInfo("FLOAT", inputShape);

  auto a = builder->addInputTensor(
      inputInfo, {TileSet::IO, ExchangeStrategy::OverlapInnerLoop});
  auto b = builder->addInputTensor(
      inputInfo, {TileSet::IO, ExchangeStrategy::OverlapInnerLoop});
  auto c = builder->addInputTensor(
      inputInfo, {TileSet::IO, ExchangeStrategy::OverlapInnerLoop});
  auto x = aiOnnx.add({a, b});
  x      = aiOnnx.add({x, c});
  builder->addOutputTensor(x);

  auto proto    = builder->getModelProto();
  auto dataFlow = DataFlow(5, {{x, AnchorReturnType("All")}});

  SessionOptions opts;
  opts.virtualGraphMode        = VirtualGraphMode::Auto;
  opts.enableExplicitMainLoops = true;
  opts.useHostCopyOps          = true;
  opts.numIOTiles              = 32;
  opts.experimentalSettings.createHostTransferableTensorWithOffset = true;

  auto device = createTestDevice(TEST_TARGET, 1);

  auto session = popart::InferenceSession::createFromOnnxModel(
      proto,
      dataFlow,
      device,
      InputShapeInfo(),
      opts,
      popart::Patterns(PatternsLevel::Default));

  session->prepareDevice();

  using Mapping = poplar::Graph::TileToTensorMapping;

  auto getStartTile = [&](const Mapping &ans) {
    unsigned index = 0;
    for (unsigned i = 0; i < ans.size(); ++i) {
      if (!ans[i].empty()) {
        index = i;
        break;
      }
    }
    return index;
  };

  std::map<std::string, unsigned> startMappings;
  auto &irLowering = session->getDevice().lowering();
  const auto &ir   = irLowering.ir();
  for (auto &id : ir.getAllTensorIds()) {
    auto *t = ir.getTensor(id);
    if (t->isHostLoadTensor()) {
      auto vgid      = t->getVirtualGraphIdAndTileSetUnsafe();
      auto &graph    = irLowering.getVirtualGraph(vgid.first, vgid.second);
      auto &tensor   = irLowering.tensors().get(t->id);
      const auto &tm = graph.getTileMapping(tensor);
      auto startTile = getStartTile(graph.getTileMapping(tensor));
      startMappings[t->id] = startTile;
      std::cout << t->id << " : " << tm << std::endl;
    }
  }

  std::set<unsigned> uniqueMappings;
  for (const auto &mappings : startMappings) {
    BOOST_CHECK(uniqueMappings.insert(mappings.second).second == true);
  }
  BOOST_CHECK(uniqueMappings.size() == startMappings.size());
}
