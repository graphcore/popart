#ifndef GUARD_NEURALNET_POPART_DOTVISUALIZER_HPP
#define GUARD_NEURALNET_POPART_DOTVISUALIZER_HPP

#include <popart/names.hpp>
#include <popart/optionflags.hpp>

namespace popart {
class Ir;
class DotVisualizer {
public:
  DotVisualizer(const Ir *, DotCheck);

  // write .dot files
  void write();

private:
  const Ir *ir;
  DotCheck check;

  using AbridgedGraphName = std::string;
  using FullGraphName     = std::string;

  // map from a GraphId to a shorter string for compactness
  std::map<FullGraphName, AbridgedGraphName> graphMapping;
  AbridgedGraphName getAbridgedGraphName(const FullGraphName &gString);

  // Each graph has it's own ofstream (if separateCallOpPdfs is true)
  std::map<AbridgedGraphName, std::ofstream> ofstreams;
  std::ofstream &strm(const FullGraphName &gString);

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

  std::set<DotCheck> getDotChecks();
};
} // namespace popart

#endif
