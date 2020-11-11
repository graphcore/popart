// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_ATTRIBUTES_HPP
#define GUARD_NEURALNET_ATTRIBUTES_HPP

#include <map>
#include <string>
#include <vector>
#include <popart/names.hpp>

namespace popart {

// Wrapper around the container of ONNX_NAMESPACE::AtrributeProtos
// of a Node, provides faster and cleaner reads of values
// from keys (strings) than ONNX_NAMESPACE::AttributesProto
class Attributes {
public:
  // The types of attributes as defined in the onnx spec
  using Ints    = std::vector<int64_t>;
  using Int     = int64_t;
  using Floats  = std::vector<float>;
  using Float   = float;
  using Strings = std::vector<std::string>;
  using String  = std::string;
  using Graph   = ONNX_NAMESPACE::GraphProto;

  Attributes(const NodeAttributes &);
  Attributes() = default;

  const std::vector<std::string> &getNames() const;
  onnxAttPtr at(const std::string &name) const;
  void append(std::stringstream &ss, std::string prefix = "") const;
  template <typename T> void setIfPresent(T &, const std::string &key) const;
  // as above, but throws an error if key not present
  template <typename T> void set(T &, const std::string &key) const;
  bool hasAttribute(const std::string &key) const;

  // Take an attribute identified by `key` from the given `Attributes` object
  void takeAttribute(const std::string &key, const Attributes &attributes);

  // Take the set of attributes that match the given predicate
  template <typename UnaryPredicate> Attributes filter(UnaryPredicate p) const {
    Attributes result;

    for (auto &name : names) {
      if (p(name)) {
        result.takeAttribute(name, *this);
      }
    }

    return result;
  }

  // Attempt to get the value of attribute assigned to the key, else return the
  // the default value
  template <typename T>
  T getAttribute(const std::string &key, const T &defaultValue) const;

  // Attempt to get the value of attribute assigned to the key, else throw an
  // exception
  template <typename T> T getAttribute(const std::string &key) const;

  // Adds the key and value
  template <typename T> void setAttribute(const std::string &key, T &);

private:
  std::map<std::string, onnxAttPtr> att_map;
  std::vector<std::string> names;
};

// Check that `x` is 0 or 1 before doing the conversion to bool.
// This is common when using attributes as onnx uses int64 for boolean values.
bool checkedIntToBool(int64_t x);

// Template specialisation that allows a string to be used like a predicate
template <> Attributes Attributes::filter(const char *key) const;
template <> Attributes Attributes::filter(std::string key) const;

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

template <>
void Attributes::set(std::vector<float> &vs, const std::string &key) const;

template <>
void Attributes::set(std::vector<std::string> &vs,
                     const std::string &key) const;

template <> void Attributes::set(float &v, const std::string &key) const;
template <> void Attributes::set(int64_t &v, const std::string &key) const;

template <>
Attributes::Ints
Attributes::getAttribute(const std::string &key,
                         const Attributes::Ints &defaultValue) const;
template <>
Attributes::Int
Attributes::getAttribute(const std::string &key,
                         const Attributes::Int &defaultValue) const;
template <>
Attributes::String
Attributes::getAttribute(const std::string &key,
                         const Attributes::String &defaultValue) const;
template <>
Attributes::Float
Attributes::getAttribute(const std::string &key,
                         const Attributes::Float &defaultValue) const;

template <>
Attributes::Ints Attributes::getAttribute(const std::string &key) const;
template <>
Attributes::Int Attributes::getAttribute(const std::string &key) const;
template <>
Attributes::String Attributes::getAttribute(const std::string &key) const;
template <>
Attributes::Strings Attributes::getAttribute(const std::string &key) const;
template <>
Attributes::Float Attributes::getAttribute(const std::string &key) const;
template <>
Attributes::Floats Attributes::getAttribute(const std::string &key) const;
template <>
Attributes::Graph Attributes::getAttribute(const std::string &key) const;

template <>
void Attributes::setAttribute(const std::string &key, Attributes::Ints &);
template <>
void Attributes::setAttribute(const std::string &key, Attributes::Int &);
template <>
void Attributes::setAttribute(const std::string &key, Attributes::String &);
} // namespace popart

#endif
