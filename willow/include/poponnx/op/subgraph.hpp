#ifndef GUARD_NEURALNET_SUBGRAPH_HPP
#define GUARD_NEURALNET_SUBGRAPH_HPP

#include <poponnx/op.hpp>

namespace poponnx {

class SubgraphOp : public Op {

public:
  SubgraphOp(Ir &ir_, int64_t cacheId_);

  // Book keeping structure to understand the mapping of input/outs of the child
  // op
  struct OpInfo {

    struct TensorInfo {
      TensorId id;
      bool external; // If this tensor is external to the subgraph
    };

    Op *op = nullptr;
    std::map<InIndex, TensorInfo> inputs; // Information about the op's inputs
    std::map<OutIndex, TensorInfo>
        outputs; // Information about the op's outputs
  };

  int64_t getSubgraphId() { return cacheId; }

  std::vector<OpInfo> &getChildOpsInfo() { return childOpsInfo; }
  std::vector<Op *> getOps();

  void appendAttributes(OpSerialiserBase &) const override;

private:
  std::vector<OpInfo> childOpsInfo;
  int64_t cacheId;
};

} // namespace poponnx

#endif
