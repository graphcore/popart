// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <cstdint>
#include <onnx/onnx_pb.h>
#include <onnxpasses/suffixer.hpp>
#include <vector>

#include "onnxpasses/onnxnames.hpp"

namespace popart {
namespace onnxpasses {

using namespace ONNX_NAMESPACE;

Suffixer::Suffixer(const GraphProto &g) {

  // Gather all Tensor names in GraphProto. Inputs and NodeProto outputs.
  std::vector<std::string> names;
  names.reserve(static_cast<uint64_t>(g.input_size() + g.node_size()));
  for (const auto &i : g.input()) {
    names.push_back(i.name());
  }
  for (auto x : g.node()) {
    names.insert(names.end(), x.output().cbegin(), x.output().cend());
  }

  // try  finding this string in a Tensor's name:
  //   _onnxtoonnxd_
  //
  // if found, add a '_' to the end and try again. Repeat until not found.
  // _onnxtoonnxd__, _onnxtoonnxd___, etc.
  //
  base           = "_onnxtoonnxd";
  bool baseFound = true;
  while (baseFound) {
    base += "_";
    baseFound = false;
    for (const auto &n : names) {
      if (n.find(base) != std::string::npos) {
        baseFound = true;
        break;
      }
    }
  }
}

/** Get a unique base, and increment the counter */
std::string Suffixer::getAndIncr() {
  // We use a mutex in case this class is used on multiple threads,
  // future-proofing in the unlikely event that this happens...
  std::lock_guard<std::mutex> a(m);
  auto name = base + std::to_string(n);
  ++n;
  return name;
}
} // namespace onnxpasses
} // namespace popart
