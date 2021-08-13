// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_DynamicOpTransform_HPP
#define GUARD_NEURALNET_DynamicOpTransform_HPP

#include <string>
#include <vector>

#include <popart/transforms/transform.hpp>

namespace popart {

class AliasModel;
class Op;

class DynamicOpTransform : public Transform {
public:
  static std::size_t id();

  DynamicOpTransform() : Transform() {}
  virtual ~DynamicOpTransform() override {}

  virtual bool apply(Graph &graph) const override final;

  virtual std::size_t getId() const override final { return id(); }

  void transferProperties(Op *from, Op *to) const;
  void inplace(Op *from) const;

  virtual std::string getName() const override final {
    return "DynamicOpTransform";
  }

private:
  // Turns DynamicUpdateInplaceOps and DynamicAddInplaceOp, used to form
  // gradients and possibly summed together, into a chain of updates. These ops
  // originate from converted DynamicSlicePadGradOps.
  void chainDynamicInplaceGradOps(
      Ir &ir,
      const std::map<Op *, std::vector<Op *>, POpCmp> &opsToChainMap,
      AliasModel &aliasModel) const;

  // Turns a gradient sum of one more more DynamicUpdateInplaceOp/
  // DynamicAddInplaceOps into a chain of updating a constant (zero) tensor or
  // existing sum result
  void gradSumToGradChain(Ir &ir,
                          Op *sumOp,
                          std::vector<Op *> dynamicOps,
                          AliasModel &aliasModel) const;
};

} // namespace popart

#endif
