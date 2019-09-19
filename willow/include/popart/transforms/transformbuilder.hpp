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

  TensorId getNextId(const std::string &name, OutIndex n = 0);

  TensorId concat(std::vector<TensorId> &inputs,
                  boost::optional<int64_t> virtualGraphId,
                  boost::optional<int64_t> pipelineStage,
                  const std::string opName,
                  const std::string outputName);

  TensorId matmul(TensorId lhs,
                  TensorId rhs,
                  boost::optional<int64_t> virtualGraphId,
                  boost::optional<int64_t> pipelineStage,
                  const std::string opName,
                  const std::string outputName);

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
};

} // namespace popart

#endif
