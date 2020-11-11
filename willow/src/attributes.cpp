// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <memory>
#include <onnx/onnx_pb.h>
#include <sstream>
#include <popart/attributes.hpp>
#include <popart/error.hpp>
#include <popart/util.hpp>

namespace popart {

bool checkedIntToBool(int64_t x) {
  if (x != 0 && x != 1) {
    throw error("Can not convert int {} to bool. Value needs to be 0 or 1", x);
  }
  return static_cast<bool>(x);
}

const std::vector<std::string> &Attributes::getNames() const { return names; }

onnxAttPtr Attributes::at(const std::string &name) const {
  return att_map.at(name);
}

template <>
void Attributes::setIfPresent(int64_t &v, const std::string &s) const {
  auto found = att_map.find(s);
  if (found != att_map.end()) {
    v = found->second->i();
  }
}

template <> void Attributes::setIfPresent(bool &v, const std::string &s) const {
  auto found = att_map.find(s);
  if (found != att_map.end()) {
    v = found->second->i() != 0;
  }
}

template <> void Attributes::setIfPresent(int &v, const std::string &s) const {
  auto found = att_map.find(s);
  if (found != att_map.end()) {
    v = found->second->i() != 0;
  }
}

template <>
void Attributes::setIfPresent(std::string &v, const std::string &s) const {
  auto found = att_map.find(s);
  if (found != att_map.end()) {

    // An empty string is treated as not present
    if (found->second->s() != "") {
      v = found->second->s();
    }
  }
}

template <>
void Attributes::setIfPresent(float &v, const std::string &s) const {
  auto found = att_map.find(s);
  if (found != att_map.end()) {
    v = found->second->f();
  }
}

template <>
void Attributes::setIfPresent(std::vector<int64_t> &vs,
                              const std::string &s) const {
  auto found = att_map.find(s);
  if (found != att_map.end()) {
    vs.resize(0);
    vs.reserve(found->second->ints_size());
    for (auto &v : found->second->ints()) {
      vs.push_back(v);
    }
  }
}

template <>
void Attributes::set(std::vector<int64_t> &vs, const std::string &key) const {
  auto found = att_map.find(key);
  if (found != att_map.end()) {
    vs.resize(0);
    vs.reserve(found->second->ints_size());
    for (auto &v : found->second->ints()) {
      vs.push_back(v);
    }
  } else {
    throw error("no attribute key {}", key);
  }
}

template <>
void Attributes::set(std::vector<float> &vs, const std::string &key) const {
  auto found = att_map.find(key);
  if (found != att_map.end()) {
    vs.resize(0);
    vs.reserve(found->second->floats_size());
    for (auto &v : found->second->floats()) {
      vs.push_back(v);
    }
  } else {
    throw error("no attribute key {}", key);
  }
}

template <>
void Attributes::set(std::vector<std::string> &vs,
                     const std::string &key) const {
  auto found = att_map.find(key);
  if (found != att_map.end()) {
    vs.resize(0);
    vs.reserve(found->second->strings_size());
    for (auto &v : found->second->strings()) {
      vs.push_back(v);
    }
  } else {
    throw error("no attribute key {}", key);
  }
}

template <> void Attributes::set(int64_t &v, const std::string &key) const {
  auto found = att_map.find(key);
  if (found != att_map.end()) {
    v = static_cast<int64_t>(found->second->i());
  } else {
    throw error("no attribute key {}", key);
  }
}

template <> void Attributes::set(float &v, const std::string &key) const {
  auto found = att_map.find(key);
  if (found != att_map.end()) {
    v = found->second->f();
  } else {
    throw error("no attribute key {}", key);
  }
}

template <> void Attributes::set(std::string &v, const std::string &key) const {
  auto found = att_map.find(key);
  if (found != att_map.end()) {
    v = found->second->s();
  } else {
    throw error("no attribute key {}", key);
  }
}

bool Attributes::hasAttribute(const std::string &key) const {
  return att_map.count(key) > 0;
}

void Attributes::takeAttribute(const std::string &key,
                               const Attributes &attributes) {
  if (attributes.hasAttribute(key)) {
    names.push_back(key);
    att_map[key] = attributes.att_map.at(key);
  }
}

template <> Attributes Attributes::filter(const char *key) const {
  return filter(std::string(key));
}

template <> Attributes Attributes::filter(std::string key) const {
  return filter([&](const std::string &name) { return name == key; });
}

Attributes::Attributes(const NodeAttributes &attributes) {
  for (auto &attribute : attributes) {
    auto name = attribute.name();
    names.push_back(name);
    att_map[name] = &attribute;
  }
}

void Attributes::append(std::stringstream &ss, std::string prefix) const {
  using AttPro = ONNX_NAMESPACE::AttributeProto;

  std::size_t max_attr_length = 0;
  for (auto &name : names) {
    max_attr_length = std::max(max_attr_length, name.length());
  }

  for (auto &name : names) {
    ss << prefix;
    ss << padded(name, static_cast<int>(max_attr_length + 1));
    auto attptr = att_map.at(name);
    switch (attptr->type()) {
    case AttPro::UNDEFINED: {
      break;
    }
    case AttPro::FLOAT: {
      ss << attptr->f();
      break;
    }
    case AttPro::INT: {
      ss << attptr->i();
      break;
    }
    case AttPro::STRING: {
      ss << attptr->s();
      break;
    }
    case AttPro::TENSOR: {
      break;
    }
    case AttPro::GRAPH: {
      break;
    }
    case AttPro::FLOATS: {
      appendSequence(ss, attptr->floats());
      break;
    }
    case AttPro::INTS: {
      appendSequence(ss, attptr->ints());
      break;
    }
    case AttPro::STRINGS: {
      appendSequence(ss, attptr->strings());
      break;
    }
    case AttPro::TENSORS: {
      break;
    }
    case AttPro::GRAPHS: {
      break;
    }
    case AttPro::SPARSE_TENSOR: {
      ss << "(spare tensor placeholder)";
      break;
    }

    case AttPro::SPARSE_TENSORS: {
      ss << "(spare tensors placeholder)";
      break;
    }
    }

    ss << "\n";
  }
}

template <>
Attributes::Ints
Attributes::getAttribute(const std::string &key,
                         const Attributes::Ints &defaultValue) const {
  auto found = att_map.find(key);
  if (found != att_map.end()) {
    Attributes::Ints vs;
    vs.resize(0);
    vs.reserve(found->second->ints_size());
    for (auto &v : found->second->ints()) {
      vs.push_back(v);
    }
    return vs;
  }
  return defaultValue;
}
template <>
Attributes::Int
Attributes::getAttribute(const std::string &key,
                         const Attributes::Int &defaultValue) const {
  auto found = att_map.find(key);
  if (found != att_map.end()) {
    return found->second->i();
  }

  return defaultValue;
}
template <>
Attributes::String
Attributes::getAttribute(const std::string &key,
                         const Attributes::String &defaultValue) const {
  auto found = att_map.find(key);
  if (found != att_map.end()) {

    // An empty string is treated as not present
    if (found->second->s() != "") {
      return found->second->s();
    }
  }

  return defaultValue;
}
template <>
Attributes::Float
Attributes::getAttribute(const std::string &key,
                         const Attributes::Float &defaultValue) const {
  auto found = att_map.find(key);
  if (found != att_map.end()) {
    return found->second->f();
  }

  return defaultValue;
}

template <>
Attributes::Ints Attributes::getAttribute(const std::string &key) const {
  auto found = att_map.find(key);
  if (found != att_map.end()) {
    Attributes::Ints vs;
    vs.resize(0);
    vs.reserve(found->second->ints_size());
    for (auto &v : found->second->ints()) {
      vs.push_back(v);
    }
    return vs;
  }

  throw error("no attribute key {}", key);
}
template <>
Attributes::Int Attributes::getAttribute(const std::string &key) const {
  auto found = att_map.find(key);
  if (found != att_map.end()) {
    return found->second->i();
  }

  throw error("no attribute key {}", key);
}
template <>
Attributes::String Attributes::getAttribute(const std::string &key) const {
  auto found = att_map.find(key);
  if (found != att_map.end()) {

    // An empty string is treated as not present
    if (found->second->s() != "") {
      return found->second->s();
    }
  }

  throw error("no attribute key {}", key);
}
template <>
Attributes::Strings Attributes::getAttribute(const std::string &key) const {
  auto found = att_map.find(key);
  if (found != att_map.end()) {
    Attributes::Strings vs;
    vs.resize(0);
    vs.reserve(found->second->strings_size());
    for (auto &v : found->second->strings()) {
      vs.push_back(v);
    }
    return vs;
  }

  throw error("no attribute key {}", key);
}
template <>
Attributes::Float Attributes::getAttribute(const std::string &key) const {
  auto found = att_map.find(key);
  if (found != att_map.end()) {
    return found->second->f();
  }

  throw error("no attribute key {}", key);
}
template <>
Attributes::Floats Attributes::getAttribute(const std::string &key) const {
  auto found = att_map.find(key);
  if (found != att_map.end()) {
    Attributes::Floats vs;
    vs.resize(0);
    vs.reserve(found->second->floats_size());
    for (auto &v : found->second->floats()) {
      vs.push_back(v);
    }
    return vs;
  }

  throw error("no attribute key {}", key);
}
template <>
Attributes::Graph Attributes::getAttribute(const std::string &key) const {
  auto found = att_map.find(key);
  if (found != att_map.end()) {
    return found->second->g();
  }

  throw error("no attribute key {}", key);
}

// Adds the key and value
template <>
void Attributes::setAttribute(const std::string &key, Attributes::Int &value) {
  if (hasAttribute(key)) {
    set(value, key);
  } else {
    names.push_back(key);
    ONNX_NAMESPACE::AttributeProto *attribute =
        new ONNX_NAMESPACE::AttributeProto();
    attribute->set_name(key);
    attribute->set_i(value);
    att_map[key] = attribute;
  }
}

template <>
void Attributes::setAttribute(const std::string &key, Attributes::Ints &value) {
  if (hasAttribute(key)) {
    set(value, key);
  } else {
    names.push_back(key);
    ONNX_NAMESPACE::AttributeProto *attribute =
        new ONNX_NAMESPACE::AttributeProto();
    attribute->set_name(key);
    for (int i = 0; i < value.size(); ++i) {
      attribute->add_ints(value[i]);
    }
    att_map[key] = attribute;
  }
}

template <>
void Attributes::setAttribute(const std::string &key, std::string &value) {
  if (hasAttribute(key)) {
    set(value, key);
  } else {
    names.push_back(key);
    ONNX_NAMESPACE::AttributeProto *attribute =
        new ONNX_NAMESPACE::AttributeProto();
    attribute->set_name(key);
    attribute->set_s(value.c_str());
    att_map[key] = attribute;
  }
}

} // namespace popart
