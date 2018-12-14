#ifndef GUARD_NEURALNET_ATTRIBUTES_HPP
#define GUARD_NEURALNET_ATTRIBUTES_HPP

#include <map>
#include <vector>
#include <poponnx/names.hpp>

namespace poponnx {

// Wrapper around the container of onnx::AtrributeProtos
// of a Node, provides faster and cleaner reads of values
// from keys (strings) than onnx::AttributesProto
class Attributes {
public:
  Attributes(const NodeAttributes &);
  Attributes() = default;
  const std::vector<std::string> &getNames() const;
  onnxAttPtr at(const std::string &name) const;
  void append(std::stringstream &ss) const;
  template <typename T> void setIfPresent(T &, const std::string &key) const;
  // as above, but throws an error if key not present
  template <typename T> void set(T &, const std::string &key) const;
  bool hasAttribute(const std::string &key) const;

private:
  std::map<std::string, onnxAttPtr> att_map;
  std::vector<std::string> names;
};

template <>
void Attributes::setIfPresent(std::vector<int64_t> &,
                              const std::string &key) const;
template <>
void Attributes::setIfPresent(int64_t &, const std::string &key) const;
template <>
void Attributes::setIfPresent(bool &v, const std::string &key) const;
template <>
void Attributes::setIfPresent(std::string &, const std::string &key) const;
template <>
void Attributes::setIfPresent(float &, const std::string &key) const;

template <>
void Attributes::set(std::vector<int64_t> &vs, const std::string &key) const;
template <> void Attributes::set(float &v, const std::string &key) const;
template <> void Attributes::set(int64_t &v, const std::string &key) const;

} // namespace poponnx

#endif
