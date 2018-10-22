#ifndef GUARD_NEURALNET_OPTIMIZER_HPP
#define GUARD_NEURALNET_OPTIMIZER_HPP

#include <willow/names.hpp>
#include <willow/tensorinfo.hpp>

namespace willow {

// get the learning rate Tensor's id.
// Of course, the tensor is rank 0
// This function is pure string manipulation
TensorId getLearningRateId();

class Optimizer {
public:
  virtual ~Optimizer()         = default;
  Optimizer()                  = default;
  Optimizer(const Optimizer &) = default;
  // The information for all optimizer specific tensors
  virtual std::map<TensorId, TensorInfo> tensorInfos() const = 0;
  virtual std::unique_ptr<Optimizer> clone() const           = 0;
  // create an Op of the relevant type
  virtual std::unique_ptr<Op> createOp(TensorId varId, Ir *) const = 0;
  // what are the correct input names to the Op created above?
  // the names depend on the name of the Variable being updated.
  virtual std::vector<TensorId> getInputIds(TensorId varId) const = 0;
};

class BaseSGD : public Optimizer {
public:
  BaseSGD(float lr);
  float learnRate() const;

private:
  float learnRate_;
};

class SGD : public BaseSGD {
public:
  SGD(float lr);
  virtual std::map<TensorId, TensorInfo> tensorInfos() const override final;
  virtual std::unique_ptr<Optimizer> clone() const override final;
  virtual std::unique_ptr<Op> createOp(TensorId, Ir *) const override final;
  virtual std::vector<TensorId> getInputIds(TensorId) const override final;
};

// may not change during training
class ConstSGD : public BaseSGD {
public:
  ConstSGD(float lr);
  virtual std::unique_ptr<Optimizer> clone() const override final;
  virtual std::map<TensorId, TensorInfo> tensorInfos() const override final;
  virtual std::unique_ptr<Op> createOp(TensorId, Ir *) const override final;
  virtual std::vector<TensorId> getInputIds(TensorId) const override final;
};

} // namespace willow

#endif
