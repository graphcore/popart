#ifndef GUARD_NEURALNET_OPTIMIZER_HPP
#define GUARD_NEURALNET_OPTIMIZER_HPP

#include <poponnx/names.hpp>
#include <poponnx/tensorinfo.hpp>

namespace poponnx {

// get the learning rate Tensor's id.
// Of course, the tensor is rank 0
// This function is pure string manipulation
TensorId getLearningRateId(DataType dtype);

// same for weight decay tensor id
TensorId getWeightDecayId(DataType dtype);

enum class OptimizerType { SGD = 0, CONSTSGD };

class Optimizer {
public:
  virtual ~Optimizer();
  Optimizer();
  Optimizer(const Optimizer &);
  // The information for all optimizer specific tensors
  virtual std::map<TensorId, TensorInfo> tensorInfos() const = 0;
  virtual std::unique_ptr<Optimizer> clone() const           = 0;
  // create an Op of the relevant type
  virtual std::unique_ptr<Op> createOp(TensorId varId, Graph &) const = 0;
  // what are the correct input names to the Op created above?
  // the names depend on the name of the Variable being updated.
  virtual std::vector<TensorId> getInputIds(TensorId varId,
                                            DataType varType) const = 0;
  // Can this optimizer be replaced by other? This is not true
  // if for example this has no momentum by other does, as the
  // graph structure would need to change.
  virtual bool validReplacement(const Optimizer *other) const = 0;
  virtual OptimizerType type() const                          = 0;
  virtual std::string type_s() const                          = 0;
  // for all Tensors in tensorInfos, find the tensor in
  // the Graph and reset its TensorData accordingly.
  virtual void resetTensorDatas(Graph &) const = 0;
  virtual void setTensorData(Tensor *) const   = 0;
};

class BaseSGD : public Optimizer {
public:
  BaseSGD(float lr, float wd);
  float learnRate() const;
  float weightDecay() const;

private:
  float learnRate_;
  float weightDecay_;
  // We will add momentum here
};

// learning rate, weight decay and momentum may
// change during training
class SGD : public BaseSGD {
public:
  SGD(float lr, float wd = 0); // weight decay is off by default
  std::unique_ptr<Optimizer> clone() const final;
  std::map<TensorId, TensorInfo> tensorInfos() const final;
  std::unique_ptr<Op> createOp(TensorId, Graph &) const final;
  std::vector<TensorId> getInputIds(TensorId, DataType) const final;
  bool validReplacement(const Optimizer *other) const final;
  OptimizerType type() const final;
  std::string type_s() const final;
  void setTensorData(Tensor *) const final;
  void resetTensorDatas(Graph &) const final;
};

// learning rate, weight decay and momentum may not
// change during training
class ConstSGD : public BaseSGD {
public:
  ConstSGD(float lr, float wd = 0); // weight decay is off by default
  std::unique_ptr<Optimizer> clone() const final;
  std::map<TensorId, TensorInfo> tensorInfos() const final;
  std::unique_ptr<Op> createOp(TensorId, Graph &) const final;
  std::vector<TensorId> getInputIds(TensorId, DataType) const final;
  bool validReplacement(const Optimizer *other) const final;
  OptimizerType type() const final;
  std::string type_s() const final;
  void setTensorData(Tensor *) const final;
  void resetTensorDatas(Graph &) const final;
};

} // namespace poponnx

#endif
