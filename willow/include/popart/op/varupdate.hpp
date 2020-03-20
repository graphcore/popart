// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_VARUPDATE_HPP
#define GUARD_NEURALNET_VARUPDATE_HPP

#include <popart/names.hpp>
#include <popart/op.hpp>
#include <popart/optimizervalue.hpp>

namespace popart {

class VarUpdateOp : public Op {
public:
  VarUpdateOp(const OperatorIdentifier &,
              const TensorId &varId_,
              const Op::Settings &);

  // the Var to be updated received at this index
  static InIndex getVarToUpdateInIndex() { return 0; }

  // Return (a reference to) the updated Var at this index
  static OutIndex getUpdatedVarOutIndex() { return 0; }

  // This Op aliases and modifies the input at index getVarIndex()
  view::Regions aliases(InIndex in, OutIndex) const final;
  view::Regions modifies(InIndex) const final;

  const TensorId &getVarId() const { return varId; }

  // Return a map of all optimizer specific input Tensors (learning rate, etc)
  virtual std::map<InIndex, TensorId> optimizerInputs() const = 0;

  // Create a clone, but with a new name
  virtual std::unique_ptr<Op> cloneWithNewName(const TensorId &) const = 0;

private:
  TensorId varId;
};

class VarUpdateWithUpdaterOp : public VarUpdateOp {
public:
  VarUpdateWithUpdaterOp(const OperatorIdentifier &opid,
                         const TensorId &varId,
                         const Op::Settings &settings_);

  // the gradient (for SGD) or source of copy (for CopyVarUpdate) or any other
  // tensor used to update the variable tensor is received at this index
  static InIndex getUpdaterInIndex() { return 1; }
  void setup() final;
};

class VarUpdateWithoutUpdaterOp : public VarUpdateOp {
public:
  VarUpdateWithoutUpdaterOp(const OperatorIdentifier &opid_,
                            const TensorId &varId_,
                            const Op::Settings &settings_)
      : VarUpdateOp(opid_, varId_, settings_) {}
  void setup() final;
};

} // namespace popart

#endif
