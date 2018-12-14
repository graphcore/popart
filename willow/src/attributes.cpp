#include <onnx/onnx_pb.h>
#include <sstream>
#include <poponnx/attributes.hpp>
#include <poponnx/error.hpp>
#include <poponnx/util.hpp>

namespace poponnx {

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
    v = found->second->s();
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

template <> void Attributes::set(int64_t &v, const std::string &key) const {
  auto found = att_map.find(key);
  if (found != att_map.end()) {
    v = static_cast<int64_t>(found->second->i());
  } else {
    throw error("no attribute key {}", key);
  }
}

template <> void Attributes::set(int &v, const std::string &key) const {
  auto found = att_map.find(key);
  if (found != att_map.end()) {
    v = static_cast<int>(found->second->i());
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

bool Attributes::hasAttribute(const std::string &key) const {
  return att_map.count(key) > 0;
}

Attributes::Attributes(const NodeAttributes &attributes) {
  for (auto &attribute : attributes) {
    auto name = attribute.name();
    names.push_back(name);
    att_map[name] = &attribute;
  }
}

void Attributes::append(std::stringstream &ss) const {
  using AttPro = onnx::AttributeProto;
  for (auto &name : names) {
    ss << '\n';
    ss << "  " << name << "  ";
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
    }
  }
}

} // namespace poponnx
