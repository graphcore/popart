#ifndef NEURALNET_TENSOR_HPP
#define NEURALNET_TENSOR_HPP

#include <map>
#include <memory>
#include <poponnx/error.hpp>
#include <poponnx/names.hpp>
#include <poponnx/tensordata.hpp>
#include <poponnx/tensorinfo.hpp>
#include <poponnx/vertex.hpp>

namespace poponnx {

enum class TensorType {
  ActGrad = 0, // an activation or a gradient, basically any output of an Op
  Const,
  Momentum,
  Stream,
  Unknown,
  Variable,
  N // number of tensor types
};

// The consumers (Ops) of a Tensor. Note that
// one Op may consume a Tensor at multiple locations.
class Consumers {
public:
  // Consumers is specific to a unique Tensor, which is stored
  // for later use in the constructor
  Consumers(Tensor *tensorConsumed_);
  // The number of times an Op consumes a Tensor,
  // returns a non-negative integer
  int n(Op *) const;
  // increment the number of times an Op consumes
  void increment(Op *);
  // decrement the number of times an Op consumes
  void decrement(Op *);
  // increment the current counts with those in this map
  void extend(const std::map<Op *, int> &);
  // return the total number of consumptions, taking
  // into account Ops which consume multiple times,
  // so the sum over consuming nodes of the number of
  // times consumed
  int getTotal() const;
  // the number of times each consumer uses the Tensor
  const std::map<Op *, int> &getMap() const;
  // the pointers to the consumers, no duplication for
  // Ops which consume multiple times
  std::vector<Op *> getOps() const;
  // append information about this object
  void append(std::stringstream &ss);

private:
  // The number of times an Op consumes the Tensor which
  // owns these Consumers
  std::map<Op *, int> consumers_m;
  Tensor *tensorConsumed;
};

class TensorTypeInfo {
public:
  TensorTypeInfo(TensorType, std::string);
  TensorType type() const;
  const std::string &type_s() const;

private:
  TensorType tensorType_;
  std::string tensor_type_;
};
const std::map<TensorType, TensorTypeInfo> &getTensorTypeInfoMap();
std::map<TensorType, TensorTypeInfo> initTensorTypeInfoMap();

class Tensor : public Vertex {
public:
  // note : producer (if there is one)
  // must be set after construction
  Tensor(TensorId, TensorType, Ir &);
  TensorId id;
  std::string str() const final { return id; }

  // ActGrad, Variable, etc:
  TensorType tensorType() const;
  const std::string &tensor_type() const;
  const TensorTypeInfo *getTensorTypeInfo() const { return tensorTypeInfo; }

  Consumers consumers;
  // shape and data type. Not to be used before inferShape of pir has run
  TensorInfo info;

  Op *getProducer() const;
  void setProducer(Op *);
  void resetProducer(Op *);
  bool hasProducer() const;
  bool hasTensorData() const;
  TensorData *tensorData();

  template <typename... Args> void setTensorData(Args &&... args) {
    // if data has already been created and had a stream
    // connected to it, changing the data will lead to
    // the stream reading from the wrong address.
    // Rather use TensorData::resetData.
    if (data_) {
      throw error("attempting to setTensorData a second time");
    }
    data_.reset(new TensorData(std::forward<Args>(args)...));
  }

  // Get all consumer ops and the producer op
  std::vector<Op *> associatedOps() const;

private:
  Ir &ir;
  Op *producer;
  const TensorTypeInfo *tensorTypeInfo;
  // c++ note : we cannot initialise this as {nullptr} with gcc
  // when using pimpl, it must be initialised in the .cpp constructor
  std::unique_ptr<TensorData> data_;
};

} // namespace poponnx

#endif
