// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_SRC_ONNXPASSES_SUFFIXER_HPP_
#define POPART_WILLOW_SRC_ONNXPASSES_SUFFIXER_HPP_

#include <mutex>
#include <onnxpasses/onnxnames.hpp>
#include <string>

namespace popart {
namespace onnxpasses {

/**
 * Unique Tensor name generator.
 *
 * Based on all Tensor names in the GraphProto, namely all the names in
 * input() and node(), determine a unique string from which to generate unique
 * suffixes. This is used in NodePatterns to ensure that no new Nodes generate
 * names which already exist.
 * */
class Suffixer {

public:
  Suffixer(const GraphProto &g);

  /** Get a unique suffix, and increment the counter */
  std::string getAndIncr();

private:
  std::mutex m;
  std::string base;
  int n{0};
};

} // namespace onnxpasses
} // namespace popart

#endif // POPART_WILLOW_SRC_ONNXPASSES_SUFFIXER_HPP_
