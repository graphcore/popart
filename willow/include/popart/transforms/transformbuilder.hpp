#ifndef GUARD_NEURALNET_TRANSFORM_BUILDER_HPP
#define GUARD_NEURALNET_TRANSFORM_BUILDER_HPP

#include <popart/graph.hpp>

#include <boost/any.hpp>

namespace popart {

class TransformBuilder {
  Graph &graph;

  std::unique_ptr<Op> createOp(const OperatorIdentifier &opid,
                               std::map<std::string, boost::any> attributes,
                               const std::string debugPrefix);

  TensorId op(const OperatorIdentifier &_opid,
              std::vector<TensorId> &inputs,
              std::map<std::string, boost::any> attributes,
              boost::optional<int64_t> virtualGraphId,
              boost::optional<int64_t> pipelineStage,
              const std::string opName,
              const std::string outputName);

  void opWithOutput(const OperatorIdentifier &_opid,
                    std::vector<TensorId> &inputs,
                    std::map<std::string, boost::any> attributes,
                    TensorId &out,
                    boost::optional<int64_t> virtualGraphId,
                    boost::optional<int64_t> pipelineStage,
                    const std::string debugPrefix);

public:
  TransformBuilder(Graph &graph_);

  Graph &getGraph() { return graph; }

  TensorId getNextId(const std::string &name, OutIndex n = 0);

  Op *getProducer(TensorId id);

  TensorId concat(std::vector<TensorId> &inputs,
                  int64_t axis,
                  boost::optional<int64_t> virtualGraphId,
                  boost::optional<int64_t> pipelineStage,
                  const std::string opName,
                  const std::string outputName);

  void concat(std::vector<TensorId> &inputs,
              int64_t axis,
              TensorId out,
              boost::optional<int64_t> virtualGraphId,
              boost::optional<int64_t> pipelineStage,
              const std::string opName);

  TensorId concat(std::vector<TensorId> &inputs,
                  boost::optional<int64_t> virtualGraphId,
                  boost::optional<int64_t> pipelineStage,
                  const std::string opName,
                  const std::string outputName) {
    return concat(inputs, 0, virtualGraphId, pipelineStage, opName, outputName);
  }

  void concat(std::vector<TensorId> &inputs,
              TensorId out,
              boost::optional<int64_t> virtualGraphId,
              boost::optional<int64_t> pipelineStage,
              const std::string opName) {
    concat(inputs, 0, out, virtualGraphId, pipelineStage, opName);
  }

  void sum(std::vector<TensorId> &inputs,
           TensorId out,
           boost::optional<int64_t> virtualGraphId,
           boost::optional<int64_t> pipelineStage,
           const std::string opName);

  TensorId addLhsInplace(std::vector<TensorId> &inputs,
                         boost::optional<int64_t> virtualGraphId,
                         boost::optional<int64_t> pipelineStage,
                         const std::string opName,
                         const std::string outputName);

  void addLhsInplace(std::vector<TensorId> &inputs,
                     TensorId out,
                     boost::optional<int64_t> virtualGraphId,
                     boost::optional<int64_t> pipelineStage,
                     const std::string opName);

  TensorId matmul(TensorId lhs,
                  TensorId rhs,
                  boost::optional<int64_t> virtualGraphId,
                  boost::optional<int64_t> pipelineStage,
                  const std::string opName,
                  const std::string outputName,
                  std::map<std::string, boost::any> attrs = {},
                  const OperatorIdentifier _opid = Onnx::Operators::MatMul_1);

  void cast(TensorId input,
            TensorId out,
            DataType type,
            boost::optional<int64_t> virtualGraphId,
            boost::optional<int64_t> pipelineStage,
            const std::string opName);

  TensorId squeeze(TensorId in,
                   const Shape &axes,
                   boost::optional<int64_t> virtualGraphId,
                   boost::optional<int64_t> pipelineStage,
                   const std::string opName,
                   const std::string outputName);

  void squeeze(TensorId in,
               const Shape &axes,
               TensorId out,
               boost::optional<int64_t> virtualGraphId,
               boost::optional<int64_t> pipelineStage,
               const std::string opName);

  TensorId slice(TensorId in,
                 const Shape &starts,
                 const Shape &ends,
                 const Shape &axes,
                 boost::optional<int64_t> virtualGraphId,
                 boost::optional<int64_t> pipelineStage,
                 const std::string opName,
                 const std::string outputName);

  TensorId sliceInPlace(TensorId in,
                        const Shape &starts,
                        const Shape &ends,
                        const Shape &axes,
                        boost::optional<int64_t> virtualGraphId,
                        boost::optional<int64_t> pipelineStage,
                        const std::string opName,
                        const std::string outputName);

  void slice(TensorId in,
             const Shape &starts,
             const Shape &ends,
             const Shape &axes,
             TensorId out,
             boost::optional<int64_t> virtualGraphId,
             boost::optional<int64_t> pipelineStage,
             const std::string opName);

  TensorId transpose(TensorId in,
                     Shape perm,
                     boost::optional<int64_t> virtualGraphId,
                     boost::optional<int64_t> pipelineStage,
                     const std::string opName,
                     const std::string outTensorName);

  void transpose(TensorId in,
                 Shape perm,
                 TensorId out,
                 boost::optional<int64_t> virtualGraphId,
                 boost::optional<int64_t> pipelineStage,
                 const std::string opName);

  TensorId reshape(TensorId in,
                   Shape shape,
                   boost::optional<int64_t> virtualGraphId,
                   boost::optional<int64_t> pipelineStage,
                   const std::string opName,
                   const std::string outputName);

  TensorId reducesum(TensorId in,
                     int64_t keepdims,
                     std::vector<int64_t> axes,
                     boost::optional<int64_t> virtualGraphId,
                     boost::optional<int64_t> pipelineStage,
                     const std::string opName,
                     const std::string outputName);
};

} // namespace popart

#endif
