// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_GRAPHID_HPP_
#define POPART_WILLOW_INCLUDE_POPART_GRAPHID_HPP_
#include <ostream>
#include <string>

namespace popart {

class GraphId {
public:
  GraphId() = delete;
  GraphId(const std::string &);

  bool operator<(const GraphId &) const;
  bool operator==(const GraphId &) const;
  bool operator!=(const GraphId &) const;

  static const GraphId &root();

  const std::string &str() const;

private:
  std::string id;
};

std::ostream &operator<<(std::ostream &, const GraphId &);

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_GRAPHID_HPP_
