#ifndef GUARD_NEURALNET_GRAPHID_HPP
#define GUARD_NEURALNET_GRAPHID_HPP
#include <string>

namespace poponnx {

class GraphId {
public:
  GraphId() = delete;
  GraphId(const std::string &);

  bool operator<(const GraphId &) const;

  static const GraphId &root();

private:
  std::string id;
};

} // namespace poponnx

#endif
