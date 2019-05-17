#ifndef GUARD_NEURALNET_OPSERIALISER_HPP
#define GUARD_NEURALNET_OPSERIALISER_HPP
#include <boost/optional.hpp>
#include <sstream>

namespace poponnx {

class Op;
class TensorIndexMap;
class Scope;

class OpSerialiserBase {
public:
  virtual ~OpSerialiserBase() {}

  virtual void appendAttribute(const std::string &, float)               = 0;
  virtual void appendAttribute(const std::string &, int64_t)             = 0;
  virtual void appendAttribute(const std::string &, uint32_t)            = 0;
  virtual void appendAttribute(const std::string &, uint64_t)            = 0;
  virtual void appendAttribute(const std::string &,
                               const std::vector<int64_t> &)             = 0;
  virtual void appendAttribute(const std::string &, const std::string &) = 0;
  virtual void appendAttribute(const std::string &,
                               boost::optional<int64_t>)                 = 0;
  virtual void appendAttribute(const std::string &, bool)                = 0;
  virtual void appendAttribute(const std::string &, const Scope &)       = 0;

  virtual void appendForwardOp(const Op *) = 0;
};

class OpSerialiser : public OpSerialiserBase {
public:
  OpSerialiser(const Op *, std::stringstream &ss_);

  void appendAttribute(const std::string &, float) override;
  void appendAttribute(const std::string &, int64_t) override;
  void appendAttribute(const std::string &, uint32_t) override;
  void appendAttribute(const std::string &, uint64_t) override;
  void appendAttribute(const std::string &,
                       const std::vector<int64_t> &) override;
  void appendAttribute(const std::string &, const std::string &) override;
  void appendAttribute(const std::string &, boost::optional<int64_t>) override;
  void appendAttribute(const std::string &, bool) override;
  void appendAttribute(const std::string &, const Scope &) override;

  virtual void appendForwardOp(const Op *) override;

private:
  template <typename T> void appendAttr(const std::string &, const T &);

private:
  std::stringstream &ss;
  const std::string tab = "    ";
};

// Creates an Id in the format:
// <domain>::<type>::<version>_<in0>,<in1>,...,<inN>_<out0>,<out1>,...,<outN>_<attr0>_<attr1>_..._<attrN>_
class OpEquivIdCreator : public OpSerialiserBase {
public:
  OpEquivIdCreator(const Op *);

  void appendAttribute(const std::string &, float) override;
  void appendAttribute(const std::string &, int64_t) override;
  void appendAttribute(const std::string &, uint32_t) override;
  void appendAttribute(const std::string &, uint64_t) override;
  void appendAttribute(const std::string &,
                       const std::vector<int64_t> &) override;
  void appendAttribute(const std::string &, const std::string &) override;
  void appendAttribute(const std::string &, boost::optional<int64_t>) override;
  void appendAttribute(const std::string &, bool) override;
  void appendAttribute(const std::string &, const Scope &) override;

  virtual void appendForwardOp(const Op *) override;

  std::string str();

private:
  template <typename T> void appendAttr(const T &);

private:
  std::stringstream ss;
  const char sep = '_';
};

template <> void OpEquivIdCreator::appendAttr(const TensorIndexMap &tmap);

} // namespace poponnx

#endif
