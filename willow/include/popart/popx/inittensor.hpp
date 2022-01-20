// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_INITTENSOR_HPP
#define GUARD_INITTENSOR_HPP

#include <set>
#include <string>
#include <vector>
#include <poplar/Program.hpp>
#include <popart/names.hpp>
#include <popart/popx/preparedtensor.hpp>

namespace popart {
namespace popx {

enum class InitMethod {
  None = 0,
  Aliasing,
  PostIrAliasing,
  Cloning,
  Creator,
  Linear
};

std::ostream &operator<<(std::ostream &os, const InitMethod &type);

class IrLowering;
class ICreatorCandidate;
using ICreatorCandidatePtr = std::shared_ptr<ICreatorCandidate>;

/**
 *  An InitTensorBase is a class describing how to initialize a tensor.
 *  Each InitTensor object has a priority that is used to compare
 *  InitTensor objects with the same dstId
 * */
class InitTensorBase {
public:
  virtual ~InitTensorBase() {}
  InitTensorBase(InitMethod method,
                 TensorId dstId,
                 RequireParallelWritable requireParallelWritable,
                 double priority);

  // Creates a Poplar tensor corresponding to dstId
  virtual bool initTensor(IrLowering &irLowering) const = 0;

  // Compares InitTensor classes by priority and string representation
  bool operator<(InitTensorBase const &rhs) const;

  // Returns the priority
  double getPriority() const { return priority; }

  // Returns the TensorId of the tensor to be created
  TensorId getDstId() const { return dstId; }

  // If the dstId is derived from exactly one srcId, this returns true
  virtual bool hasSrcId() const { return false; }

  // Returns the TensorId of the tensor the dstId tensor is to be derived from
  virtual TensorId getSrcId() const { return ""; }

  // Returns the TensorIds of all tensors that need to exist as Poplar tensors
  // before the dstId tensor can be created
  virtual std::set<TensorId> getDependsOnIds() const;

  // String representation for each InitTensor object
  std::string str() const;

protected:
  // Additional representation
  virtual std::string extraStr() const { return ""; }

  // The method identifier, used to generate a string representation for each
  // InitTensor
  InitMethod method;

  // The tensor to be created
  TensorId dstId;

  // The resulting tensor must be parallel writable
  RequireParallelWritable requireParallelWritable;

  // The priority this InitTensor object has
  double priority;
};

using InitTensorPtr = std::shared_ptr<InitTensorBase>;

struct PInitTensorCmp {
  bool operator()(InitTensorPtr const &a, InitTensorPtr const &b) const {
    return std::make_pair(a->getPriority(), b->str()) >
           std::make_pair(b->getPriority(), a->str());
  }
};

using InitTensorPtrs = std::set<InitTensorPtr, PInitTensorCmp>;

/**
 * Create tensor dstId by directly aliasing from srcId to dstId by
 * virtue of IR aliasing
 */
class InitTensorAliasing : public InitTensorBase {
public:
  InitTensorAliasing(TensorId srcId,
                     TensorId dstId,
                     RequireParallelWritable requireParallelWritable,
                     double priority);
  bool initTensor(IrLowering &irLowering) const override;
  bool hasSrcId() const override { return true; }
  TensorId getSrcId() const override { return srcId; }

private:
  TensorId srcId;
};

/**
 * Create tensor dstId by checking if post-IR aliasing can directly alias
 * from srcId to dstId
 */
class InitTensorPostIrAliasing : public InitTensorBase {
public:
  InitTensorPostIrAliasing(TensorId srcId,
                           TensorId dstId,
                           RequireParallelWritable requireParallelWritable,
                           double priority);
  bool initTensor(IrLowering &irLowering) const override;
  bool hasSrcId() const override { return true; }
  TensorId getSrcId() const override { return srcId; }

protected:
  TensorId srcId;
};

/**
 * Create tensor dstId by directly cloning from srcId to dstId
 */
class InitTensorCloning : public InitTensorBase {
public:
  InitTensorCloning(TensorId srcId,
                    TensorId dstId,
                    RequireParallelWritable requireParallelWritable,
                    double priority);
  bool initTensor(IrLowering &irLowering) const override;
  bool hasSrcId() const override { return true; }
  TensorId getSrcId() const override { return srcId; }

protected:
  TensorId srcId;
};

/**
 * Create tensor dstId by using a qualified tensor creator candidate
 */
class InitTensorCreator : public InitTensorBase {
public:
  InitTensorCreator(ICreatorCandidatePtr candidate,
                    std::set<TensorId> mustExist,
                    TensorId dstId,
                    RequireParallelWritable requireParallelWritable,
                    double priority);
  bool initTensor(IrLowering &irLowering) const override;
  std::string extraStr() const override;
  std::set<TensorId> getDependsOnIds() const override;

private:
  ICreatorCandidatePtr candidate;
  std::set<TensorId> mustExist;
};

/**
 * Create tensor dstId by linearly mapping to IPU tiles
 */
class InitTensorLinear : public InitTensorBase {
public:
  InitTensorLinear(TensorId dstId,
                   RequireParallelWritable requireParallelWritable,
                   double priority);
  bool initTensor(IrLowering &irLowering) const override;
};

} // namespace popx
} // namespace popart

#endif
