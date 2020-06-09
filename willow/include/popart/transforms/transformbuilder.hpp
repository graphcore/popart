// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_TRANSFORM_BUILDER_HPP
#define GUARD_NEURALNET_TRANSFORM_BUILDER_HPP

#include <popart/graph.hpp>

#include <popart/any.hpp>

namespace popart {

class TransformBuilder {
  Graph &graph;

  std::unique_ptr<Op> createOp(const OperatorIdentifier &opid,
                               std::map<std::string, popart::any> attributes,
                               const std::string debugPrefix);

  TensorId op(const OperatorIdentifier &_opid,
              std::vector<TensorId> &inputs,
              std::map<std::string, popart::any> attributes,
              OptionalVGraphId,
              OptionalPipelineStage,
              OptionalPingPongPhase,
              const std::string opName,
              const std::string outputName);

  void opWithOutput(const OperatorIdentifier &_opid,
                    std::vector<TensorId> &inputs,
                    std::map<std::string, popart::any> attributes,
                    TensorId &out,
                    OptionalVGraphId,
                    OptionalPipelineStage,
                    OptionalPingPongPhase,
                    const std::string debugPrefix);

  std::vector<TensorId>
  multiOutputOp(const OperatorIdentifier &_opid,
                std::vector<TensorId> &inputs,
                OutIndex numberOfOutputs,
                std::map<std::string, popart::any> attributes,
                OptionalVGraphId,
                OptionalPipelineStage,
                OptionalPingPongPhase,
                const std::string opName);

public:
  TransformBuilder(Graph &graph_);

  Graph &getGraph() { return graph; }

  TensorId getNextId(const std::string &name, OutIndex n = 0);

  Op *getProducer(TensorId id);
  bool hasProducer(TensorId id);

  TensorId concat(std::vector<TensorId> &inputs,
                  int64_t axis,
                  OptionalVGraphId virtualGraphId,
                  OptionalPipelineStage,
                  OptionalPingPongPhase,
                  const std::string opName,
                  const std::string outputName);

  void concat(std::vector<TensorId> &inputs,
              int64_t axis,
              TensorId out,
              OptionalVGraphId,
              OptionalPipelineStage,
              OptionalPingPongPhase,
              const std::string opName);

  TensorId concat(std::vector<TensorId> &inputs,
                  OptionalVGraphId virtualGraphId,
                  OptionalPipelineStage pipelineStage,
                  OptionalPingPongPhase pingPongPhase,
                  const std::string opName,
                  const std::string outputName) {
    return concat(inputs,
                  0,
                  virtualGraphId,
                  pipelineStage,
                  pingPongPhase,
                  opName,
                  outputName);
  }

  void concat(std::vector<TensorId> &inputs,
              TensorId out,
              OptionalVGraphId virtualGraphId,
              OptionalPipelineStage pipelineStage,
              OptionalPingPongPhase pingPongPhase,
              const std::string opName) {
    concat(
        inputs, 0, out, virtualGraphId, pipelineStage, pingPongPhase, opName);
  }

  void sum(std::vector<TensorId> &inputs,
           TensorId out,
           OptionalVGraphId,
           OptionalPipelineStage,
           OptionalPingPongPhase,
           const std::string opName);

  TensorId addLhsInplace(std::vector<TensorId> &inputs,
                         OptionalVGraphId,
                         OptionalPipelineStage,
                         OptionalPingPongPhase,
                         const std::string opName,
                         const std::string outputName);

  void addLhsInplace(std::vector<TensorId> &inputs,
                     TensorId out,
                     OptionalVGraphId,
                     OptionalPipelineStage,
                     OptionalPingPongPhase,
                     const std::string opName);

  TensorId matmul(TensorId lhs,
                  TensorId rhs,
                  OptionalVGraphId,
                  OptionalPipelineStage,
                  OptionalPingPongPhase,
                  const std::string opName,
                  const std::string outputName,
                  std::map<std::string, popart::any> attrs = {},
                  const OperatorIdentifier _opid = Onnx::Operators::MatMul_1);

  void cast(TensorId input,
            TensorId out,
            DataType type,
            OptionalVGraphId,
            OptionalPipelineStage,
            OptionalPingPongPhase,
            const std::string opName);

  TensorId squeeze(TensorId in,
                   const Shape &axes,
                   OptionalVGraphId,
                   OptionalPipelineStage,
                   OptionalPingPongPhase,
                   const std::string opName,
                   const std::string outputName);

  void squeeze(TensorId in,
               const Shape &axes,
               TensorId out,
               OptionalVGraphId,
               OptionalPipelineStage,
               OptionalPingPongPhase,
               const std::string opName);

  TensorId slice(TensorId in,
                 const Shape &starts,
                 const Shape &ends,
                 const Shape &axes,
                 OptionalVGraphId,
                 OptionalPipelineStage,
                 OptionalPingPongPhase,
                 const std::string opName,
                 const std::string outputName);

  TensorId sliceInPlace(TensorId in,
                        const Shape &starts,
                        const Shape &ends,
                        const Shape &axes,
                        OptionalVGraphId,
                        OptionalPipelineStage,
                        OptionalPingPongPhase,
                        const std::string opName,
                        const std::string outputName);

  void slice(TensorId in,
             const Shape &starts,
             const Shape &ends,
             const Shape &axes,
             TensorId out,
             OptionalVGraphId,
             OptionalPipelineStage,
             OptionalPingPongPhase,
             const std::string opName);

  TensorId transpose(TensorId in,
                     Shape perm,
                     OptionalVGraphId,
                     OptionalPipelineStage,
                     OptionalPingPongPhase,
                     const std::string opName,
                     const std::string outTensorName);

  void transpose(TensorId in,
                 Shape perm,
                 TensorId out,
                 OptionalVGraphId,
                 OptionalPipelineStage,
                 OptionalPingPongPhase,
                 const std::string opName);

  TensorId reshape(TensorId in,
                   Shape shape,
                   OptionalVGraphId,
                   OptionalPipelineStage,
                   OptionalPingPongPhase,
                   const std::string opName,
                   const std::string outputName);

  TensorId reducesum(TensorId in,
                     int64_t keepdims,
                     std::vector<int64_t> axes,
                     OptionalVGraphId,
                     OptionalPipelineStage,
                     OptionalPingPongPhase,
                     const std::string opName,
                     const std::string outputName);

  std::vector<TensorId> split(TensorId in,
                              int64_t axis,
                              std::vector<int64_t> splitSizes,
                              OptionalVGraphId,
                              OptionalPipelineStage,
                              OptionalPingPongPhase,
                              const std::string opName);

  TensorId add(std::vector<TensorId> &inputs,
               OptionalVGraphId,
               OptionalPipelineStage,
               OptionalPingPongPhase,
               const std::string opName,
               const std::string outputName);

  void unsqueeze(TensorId in,
                 std::vector<int64_t> axes,
                 TensorId out,
                 OptionalVGraphId,
                 OptionalPipelineStage,
                 OptionalPingPongPhase,
                 const std::string opName);
};

} // namespace popart

#endif
