// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_POPART_DOTVISUALIZER_HPP
#define GUARD_NEURALNET_POPART_DOTVISUALIZER_HPP

#include <string>
#include <popart/names.hpp>
#include <popart/sessionoptions.hpp>

namespace popart {
class Ir;
class DotVisualizer {
public:
  DotVisualizer(std::string _check_);

  // write .dot files
  void write(const Ir &ir);

private:
  std::string check;

  using AbridgedGraphName = std::string;
  using FullGraphName     = std::string;

  // map from a GraphId to a shorter string for compactness
  std::map<FullGraphName, AbridgedGraphName> graphMapping;
  AbridgedGraphName getAbridgedGraphName(const FullGraphName &gString);

  // Each graph has it's own ofstream (if separateCallOpPdfs is true)
  std::map<AbridgedGraphName, std::ofstream> ofstreams;
  std::ofstream &strm(const FullGraphName &gString, const Ir &ir);

  // the name that an Op has in the .dot file
  std::string nodeDotId(OpId) const;

  // the name that a Tensor has in the .dot file
  std::string tensorDotId(const TensorId &) const;

  std::string getTensorNodeColor(TensorType type) const;

  // Create a tensor node in .dot file, if required
  void makeNodeIfRequired(const Tensor *tensor, std::ofstream &ofs);

  // inplace Ops will have a olive green edge, others will be black
  std::string getOpNodeColor(Op *op);

  // we keep track of which tensors have been defined in the .dot file(s)
  std::set<TensorId> tensorsVisited{};

  // The schedule index within a Graph
  std::map<FullGraphName, int> graphScheduleCounter;
  int getNextGraphIndex(const FullGraphName &gString);

  std::set<std::string> getDotChecks(const Ir &ir);
};
} // namespace popart

#endif
