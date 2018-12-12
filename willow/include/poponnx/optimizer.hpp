#ifndef GUARD_NEURALNET_OPTIMIZER_HPP
#define GUARD_NEURALNET_OPTIMIZER_HPP

#include <poponnx/names.hpp>
#include <poponnx/tensorinfo.hpp>

namespace poponnx {

// get the learning rate Tensor's id.
// Of course, the tensor is rank 0
// This function is pure string manipulation
TensorId getLearningRateId();

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
  virtual std::unique_ptr<Op> createOp(TensorId varId, Ir *) const = 0;
  // what are the correct input names to the Op created above?
  // the names depend on the name of the Variable being updated.
  virtual std::vector<TensorId> getInputIds(TensorId varId) const = 0;
  // Can this optimizer be replaced by other? This is not true
  // if for example this has no momentum by other does, as the
  // graph structure would need to change.
  virtual bool validReplacement(const Optimizer *other) const = 0;
  virtual OptimizerType type() const                          = 0;
  virtual std::string type_s() const                          = 0;
  // for all Tensors in tensorInfos, find the tensor in
  // the Ir and reset its TensorData accordingly.
  virtual void resetTensorDatas(Ir *) const  = 0;
  virtual void setTensorData(Tensor *) const = 0;
};

class BaseSGD : public Optimizer {
public:
  BaseSGD(float lr);
  float learnRate() const;

private:
  float learnRate_;
  // We will add momentum here
};

// learning rate and momentum may change during training
class SGD : public BaseSGD {
public:
  SGD(float lr);
  std::unique_ptr<Optimizer> clone() const final;
  std::map<TensorId, TensorInfo> tensorInfos() const final;
  std::unique_ptr<Op> createOp(TensorId, Ir *) const final;
  std::vector<TensorId> getInputIds(TensorId) const final;
  bool validReplacement(const Optimizer *other) const final;
  OptimizerType type() const final;
  std::string type_s() const final;
  void setTensorData(Tensor *) const final;
  void resetTensorDatas(Ir *) const final;
};

// learning rate and momentum aren't allowed to change during training
class ConstSGD : public BaseSGD {
public:
  ConstSGD(float lr);
  std::unique_ptr<Optimizer> clone() const final;
  std::map<TensorId, TensorInfo> tensorInfos() const final;
  std::unique_ptr<Op> createOp(TensorId, Ir *) const final;
  std::vector<TensorId> getInputIds(TensorId) const final;
  bool validReplacement(const Optimizer *other) const final;
  OptimizerType type() const final;
  std::string type_s() const final;
  void setTensorData(Tensor *) const final;
  void resetTensorDatas(Ir *) const final;
};

} // namespace poponnx

#endif
