// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <popart/graphid.hpp>

#include <ostream>

namespace popart {

const GraphId rootId("");

GraphId::GraphId(const std::string &id_) : id(id_) {}

bool GraphId::operator<(const GraphId &other) const {
  return this->id < other.id;
}

bool GraphId::operator==(const GraphId &other) const {
  return this->id == other.id;
}

bool GraphId::operator!=(const GraphId &other) const {
  return this->id != other.id;
}

const GraphId &GraphId::root() { return rootId; }

std::string GraphId::str() const { return id; }

std::ostream &operator<<(std::ostream &ss, const GraphId &graph_id) {
  ss << graph_id.str();
  return ss;
}

} // namespace popart
