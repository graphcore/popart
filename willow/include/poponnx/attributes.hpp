#ifndef GUARD_NEURALNET_ATTRIBUTES_HPP
#define GUARD_NEURALNET_ATTRIBUTES_HPP

#include <poponnx/names.hpp>

namespace willow {

// Wrapper around the container of onnx::AtrributeProtos
// of a Node, provides faster and cleaner reads of values
// from keys (strings) than onnx::AttribtesProto
class Attributes {
public:
  Attributes(decltype(onnx::NodeProto().attribute()) &);
  Attributes() = default;
  const std::vector<std::string> &getNames() const;
  onnxAttPtr at(std::string name) const;
  void append(std::stringstream &ss) const;
  template <typename T> void setIfPresent(T &, std::string key) const;
  // as above, but throws an error if key not present
  template <typename T> void set(T &, std::string key) const;

private:
  std::map<std::string, onnxAttPtr> att_map;
  std::vector<std::string> names;
};

template <> void Attributes::setIfPresent(int64_t &, std::string s) const;

template <> void Attributes::setIfPresent(bool &v, std::string s) const;

template <>
void Attributes::setIfPresent(std::vector<int64_t> &, std::string s) const;

template <> void Attributes::setIfPresent(std::string &, std::string s) const;

template <>
void Attributes::set(std::vector<int64_t> &vs, std::string key) const;

} // namespace willow

#endif
