// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <onnxpasses/nodepatterns/constfolder.hpp>
#include <onnxpasses/nodepatterns/constfoldops.hpp>

namespace popart {
namespace onnxpasses {

using namespace ONNX_NAMESPACE;

ConstFolder::ConstFolder(std::shared_ptr<PatternTarget> t) : NodePattern(t) {
  foldConstants = constants();
  registerConstFoldOpMap();
}

ConstFolder::~ConstFolder() {}

/**
 * ConstFolder design, benefits.
 * For onnx -> onnx transformations we iterate through the nodes of graph of
 * size N and apply/match-check M node patterns on them. O(N*M) steps. We reduce
 * M by having ConstFolder as one pattern instead of separate patterns for Abs,
 * Constant, etc. by the number of constant fold ops. And we hash-map constant
 * folder subclasses AbsCFold, ConcatCFold, etc. on a node improving
 * performance.
 * */

bool ConstFolder::go(const NodeProto &node) {

  const auto type = node.op_type();
  if (constFoldOpMap.count(type)) {
    auto const input = inputReady(node);
    if (!std::get<0>(input)) {
      return false;
    }
    const auto outputs = constFoldOpMap[type]->fold(node, std::get<1>(input));
    foldConstants->insert(outputs.begin(), outputs.end());
  }

  return false;
}

std::tuple<bool, Constants>
ConstFolder::inputReady(const NodeProto &node) const {
  Constants inputs;
  for (auto i : node.input()) {
    const auto found = foldConstants->find(i);
    if (found == foldConstants->cend()) {
      return std::make_tuple(false, inputs);
    }
    inputs.insert({i, found->second});
  }
  return std::make_tuple(true, inputs);
}

void ConstFolder::registerConstFoldOpMap() {

  std::vector<std::unique_ptr<ConstFoldOp>> tmpConstFoldOps;
  tmpConstFoldOps.push_back(std::make_unique<AbsCFold>());
  tmpConstFoldOps.push_back(std::make_unique<BinaryCFold>());
  tmpConstFoldOps.push_back(std::make_unique<CastCFold>());
  tmpConstFoldOps.push_back(std::make_unique<ConcatCFold>());
  tmpConstFoldOps.push_back(std::make_unique<ConstantCFold>());
  tmpConstFoldOps.push_back(std::make_unique<ConstantOfShapeCFold>());
  tmpConstFoldOps.push_back(std::make_unique<ExpCFold>());
  tmpConstFoldOps.push_back(std::make_unique<ExpandCFold>());
  tmpConstFoldOps.push_back(std::make_unique<FloorCFold>());
  tmpConstFoldOps.push_back(std::make_unique<FmodCFold>());
  tmpConstFoldOps.push_back(std::make_unique<GatherCFold>());
  tmpConstFoldOps.push_back(std::make_unique<IdentityCFold>());
  tmpConstFoldOps.push_back(std::make_unique<IdentityLossCFold>());
  tmpConstFoldOps.push_back(std::make_unique<NegCFold>());
  tmpConstFoldOps.push_back(std::make_unique<ReciprocalCFold>());
  tmpConstFoldOps.push_back(std::make_unique<ReluCFold>());
  tmpConstFoldOps.push_back(std::make_unique<ReshapeCFold>());
  tmpConstFoldOps.push_back(std::make_unique<ScaleCFold>());
  tmpConstFoldOps.push_back(std::make_unique<ShapeCFold>());
  tmpConstFoldOps.push_back(std::make_unique<SliceCFold>());
  tmpConstFoldOps.push_back(std::make_unique<SqueezeCFold>());
  tmpConstFoldOps.push_back(std::make_unique<TransposeCFold>());
  tmpConstFoldOps.push_back(std::make_unique<UnsqueezeCFold>());

  for (int i = 0; i < tmpConstFoldOps.size(); i++) {
    std::string keyName = tmpConstFoldOps[i]->constPatternType();
    constFoldOpMap.insert(
        std::make_pair(keyName, std::move(tmpConstFoldOps[i])));
  }
}

} // namespace onnxpasses
} // namespace popart
