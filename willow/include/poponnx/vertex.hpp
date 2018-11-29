#ifndef NEURALNET_VERTEX_HPP
#define NEURALNET_VERTEX_HPP

namespace poponnx {

// All Vertices are partitioned into FWD, BWD, LOSS.
// Recompute Ops and gradient Ops are BWD
// All Loss Ops (nll, l1loss)
// and the final sum over losses are LOSS
enum class Phase { FWD = 0, BWD, LOSS, UNDEFINED };

std::map<Phase, std::string> init_phase_names();
const std::map<Phase, std::string> &phase_names();

// All vertices, except those
// in FWD on paths which don't lead to any BWD vertex
enum class PathToBwd { YES, NO, UNDEFINED };

class Vertex {

public:
  Vertex()          = default;
  virtual ~Vertex() = default;
  // The copy constructor does not copy any
  // fields, all fields are set to the "unset" value
  Vertex(const Vertex &) : Vertex() {}
  Vertex &operator=(const Vertex &) = delete;

  void incrNPathsToLoss();
  int nPathsToLoss() const;
  void setNPathsToLossToZero();

  void setPhase(Phase);
  Phase getPhase() const;

  void setPathToBwd(PathToBwd);
  bool hasPathToBwd() const;

  bool isFwdToBwd() const;

  virtual std::string str() const = 0;

private:
  static int undefinedNPaths;
  int nPathsToLoss_{undefinedNPaths};
  Phase phase_{Phase::UNDEFINED};
  PathToBwd path_to_bwd_;
};

} // namespace poponnx

#endif
