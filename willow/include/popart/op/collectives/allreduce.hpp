// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_ALLREDUCE_HPP
#define GUARD_NEURALNET_ALLREDUCE_HPP

#include <popart/op.hpp>
#include <popart/op/collectives/collectives.hpp>
#include <popart/vendored/optional.hpp>

namespace popart {

class AllReduceOp : public Op {
public:
  AllReduceOp(const OperatorIdentifier &_opid,
              CollectiveOperator op_,
              std::vector<int64_t> ipus_,
              const Op::Settings &settings_);

  AllReduceOp(const OperatorIdentifier &_opid,
              CollectiveOperator op_,
              std::vector<int64_t> ipus_,
              const bool identicalInputs_,
              const bool identicalGradInputs_,
              const Op::Settings &settings_);

  std::unique_ptr<Op> clone() const override;
  std::vector<std::unique_ptr<Op>> getGradOps() override;
  void setup() final;

  // Inputs and outputs are variadic
  static InIndex getInStartIndex() { return 0; }
  static OutIndex getOutStartIndex() { return 0; }

  void appendOutlineAttributes(OpSerialiserBase &) const override;

  bool canBeReplacedByIdentity() const override;

  float getSubgraphValue() const override { return getHighSubgraphValue(); }

  VGraphIdAndTileSet
  getIntrospectionInVirtualGraphId(InIndex index,
                                   std::set<OpId> &visited) const override;

  VGraphIdAndTileSet
  getIntrospectionOutVirtualGraphId(OutIndex index,
                                    std::set<OpId> &visited) const override;

  CollectiveOperator getReduceOp() const { return reduceOp; }

  bool getIdenticalInputs() const { return identicalInputs; }

  std::vector<int64_t> getIpus() const { return ipus; }

private:
  const CollectiveOperator reduceOp;
  const std::vector<int64_t> ipus;
  const bool identicalInputs     = false;
  const bool identicalGradInputs = false;
};

class AllReduceGradOp : public AllReduceOp {
public:
  AllReduceGradOp(CollectiveOperator op_,
                  std::vector<int64_t> ipus_,
                  const bool identicalInputs_,
                  const bool identicalGradInputs_,
                  const Op::Settings &settings_);

  std::unique_ptr<Op> clone() const override;
  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;

private:
  std::vector<GradInOutMapper> inGradMap;
  std::map<int, int> outGradMap;
};

} // namespace popart

#endif
