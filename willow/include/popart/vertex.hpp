// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef NEURALNET_VERTEX_HPP
#define NEURALNET_VERTEX_HPP

#include <string>

namespace popart {

enum class PathToLoss { Yes, No, Undefined };
enum class PathFromLoss { Yes, No, Undefined };
enum class ScheduledPreLoss { Yes, No, Undefined };

std::ostream &operator<<(std::ostream &, const PathToLoss &);
std::ostream &operator<<(std::ostream &, const PathFromLoss &);
std::ostream &operator<<(std::ostream &, const ScheduledPreLoss &);

class Vertex {

public:
  Vertex()          = default;
  virtual ~Vertex() = default;

  // The copy constructor does not copy any
  // fields, all fields are set to the "Undefined" value
  Vertex(const Vertex &) : Vertex() {}

  Vertex &operator=(const Vertex &) = delete;

  // Is there a path from this Vertex to the final loss Op? More specifically,
  // would there be such a path if the final loss Op had not been pruned.
  PathToLoss toLoss{PathToLoss::Undefined};

  // Is there a path from the final loss Op to this Vertex?
  PathFromLoss fromLoss{PathFromLoss::Undefined};

  // Is this Vertex currently scheduled before the final loss Op?
  ScheduledPreLoss scheduledPreLoss{ScheduledPreLoss::Undefined};

  int nEdgesToLoss{undefinedNEdges};

  virtual std::string str() const = 0;

  // A string summarising toLoss, fromLoss, scheduledPreLoss
  std::string wrtLossStr() const;

private:
  static constexpr int undefinedNEdges = -9;
};

} // namespace popart

#endif
