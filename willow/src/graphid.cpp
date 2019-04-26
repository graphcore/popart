#include <poponnx/graphid.hpp>

namespace poponnx {

const GraphId rootId("");

GraphId::GraphId(const std::string &id_) : id(id_) {}

bool GraphId::operator<(const GraphId &other) const {
  return this->id < other.id;
}

const GraphId &GraphId::root() { return rootId; }

} // namespace poponnx
