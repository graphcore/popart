// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_DynamicOpTransform_HPP
#define GUARD_NEURALNET_DynamicOpTransform_HPP

#include <popart/aliases.hpp>
#include <popart/op.hpp>
#include <popart/transforms/transform.hpp>

namespace popart {

class DynamicOpTransform : public Transform {
public:
  static std::size_t id();

  DynamicOpTransform() : Transform() {}
  virtual ~DynamicOpTransform() override {}

  virtual bool apply(Graph &graph) const override final;

  virtual std::size_t getId() const override final { return id(); }

  void transferProperties(Op *from, Op *to) const;
  void inplace(Op *from) const;
  void
  gradSumToGradChain(Ir &ir,
                     std::map<Op *, std::vector<Op *>, POpCmp> opsToChainMap,
                     Aliases &aliases) const;

  virtual std::string getName() const override final {
    return "DynamicOpTransform";
  }
};

} // namespace popart

#endif
