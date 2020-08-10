// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_OPSERIALISER_HPP
#define GUARD_NEURALNET_OPSERIALISER_HPP
#include <map>
#include <sstream>
#include <popart/basicoptionals.hpp>
#include <popart/names.hpp>
#include <popart/vendored/optional.hpp>

namespace popart {

class Op;
class TensorIndexMap;
class Scope;

class OpSerialiserBase {
public:
  virtual ~OpSerialiserBase() {}

  void appendAttribute(const std::string &, float);
  void appendAttribute(const std::string &, int);
  void appendAttribute(const std::string &, int64_t);
  void appendAttribute(const std::string &, uint32_t);
  void appendAttribute(const std::string &, uint64_t);
  void appendAttribute(const std::string &, const std::string &);
  void appendAttribute(const std::string &, const std::vector<int64_t> &);
  void appendAttribute(const std::string &, const Scope &);
  void appendAttribute(const std::string &, bool);

  virtual void appendAttribute(const std::string &,
                               nonstd::optional<int64_t>)          = 0;
  virtual void appendAttribute(const std::string &,
                               nonstd::optional<float>)            = 0;
  virtual void appendAttribute(const std::string &,
                               const std::map<TensorId, uint64_t>) = 0;

  // For OptionalExecutionPhase, OptionalVGraphId, etc.
  template <typename T, uint32_t V>
  void appendAttribute(const std::string &key,
                       const BasicOptional<T, V> &value) {
    std::ostringstream oss;
    oss << value;
    appendStrAttr(key, oss.str());
  }

  virtual void appendForwardOp(const Op *) = 0;

private:
  virtual void appendStrAttr(const std::string &, const std::string &value) = 0;
};

class OpSerialiser : public OpSerialiserBase {
public:
  OpSerialiser(const Op *, std::stringstream &ss_);

  void appendAttribute(const std::string &, nonstd::optional<int64_t>) override;
  void appendAttribute(const std::string &, nonstd::optional<float>) override;
  void appendAttribute(const std::string &,
                       const std::map<TensorId, uint64_t>) override;

  virtual void appendForwardOp(const Op *) override;

private:
  template <typename T> void appendAttr(const std::string &, const T &);
  void appendStrAttr(const std::string &, const std::string &value) final;
  std::stringstream &ss;
  const std::string tab = "    ";
};

class OpJsonSerialiser : public OpSerialiserBase {
public:
  OpJsonSerialiser(const Op *, std::stringstream &ss_);

  void appendAttribute(const std::string &, nonstd::optional<int64_t>) override;
  void appendAttribute(const std::string &, nonstd::optional<float>) override;
  void appendAttribute(const std::string &,
                       const std::map<TensorId, uint64_t>) override;

  virtual void appendForwardOp(const Op *) override;

private:
  template <typename T> void appendAttr(const std::string &, const T &);
  void appendStrAttr(const std::string &, const std::string &value) final;

  template <typename T>
  void appendKeyValue(const std::string key, T value, bool last = false);
  void appendKeyValueFn(const std::string key,
                        std::function<void()> func,
                        bool last = false);
  void appendKeyValues(const std::string key,
                       std::function<void()> func,
                       bool last = false);

private:
  std::stringstream &ss;
  bool attributesAppended = false;
};

// Creates an Id in the format:
// <domain>::<type>::<version>_<in0>,<in1>,...,<inN>_<out0>,<out1>,...,<outN>_<attr0>_<attr1>_..._<attrN>_
class OpEquivIdCreator : public OpSerialiserBase {
public:
  OpEquivIdCreator(const Op *);

  void appendAttribute(const std::string &, nonstd::optional<int64_t>) override;
  void appendAttribute(const std::string &, nonstd::optional<float>) override;
  void appendAttribute(const std::string &,
                       const std::map<TensorId, uint64_t>) override;

  virtual void appendForwardOp(const Op *) override;

  std::string str();

private:
  void appendStrAttr(const std::string &, const std::string &value) final;
  template <typename T> void appendAttr(const T &);

private:
  std::stringstream ss;
  const char sep = '_';
};

template <> void OpEquivIdCreator::appendAttr(const TensorIndexMap &tmap);

} // namespace popart

#endif
