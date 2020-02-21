#ifndef GUARD_NEURALNET_LOOP_HPP
#define GUARD_NEURALNET_LOOP_HPP

#include <popart/op.hpp>

namespace popart {

class LoopOp : public Op {
public:
  LoopOp(const OperatorIdentifier &,
         const Op::Settings &,
         const GraphId &,
         const std::vector<std::pair<TensorId, TensorInfo>> inputs,
         const std::vector<TensorId> implicitTensors,
         const std::vector<std::pair<TensorId, TensorInfo>> explicitTensors);

  void setup() final;
  void appendAttributes(OpSerialiserBase &) const override;
  void connectOutput(const int outIdx);
  void connectInTensor(InIndex inIndex, TensorId tensorId) final;
  float getSubgraphValue() const final { return getLowSubgraphValue(); }
  std::vector<TensorId> getInputsForGraph(const Graph &graph) const final;
  std::unique_ptr<Op> clone() const override;
  std::vector<const Graph *> getCalledGraphs() const final;
  std::vector<TensorId> implicitInputTensors() const;
  std::vector<std::pair<TensorId, TensorInfo>> inputMap() const {
    return inputs_;
  }
  std::vector<std::pair<TensorId, TensorInfo>> explicitTensorMap() const {
    return explicitTensors_;
  }

  Graph &subgraph() const;

  int32_t tripCountValue() const { return tripCountValue_; }

  static InIndex getMaximumTripCountInIndex() { return 0; }
  static InIndex getTerminationConditionInIndex() { return 1; }

private:
  int32_t tripCountValue_;
  const GraphId subgraphId_;
  std::vector<TensorId> implicitTensors_;
  std::vector<std::pair<TensorId, TensorInfo>> inputs_;
  std::vector<std::pair<TensorId, TensorInfo>> explicitTensors_;
};

} // namespace popart

#endif
