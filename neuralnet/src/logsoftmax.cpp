#include <neuralnet/logsoftmax.hpp>
#include <neuralnet/tensor.hpp>

namespace neuralnet {

LogSoftmaxOp::LogSoftmaxOp(const onnx::NodeProto &node, Graph *pgraph)
    : Op(node, pgraph) {
}

void LogSoftmaxOp::setup() { output.tensor(0)->info = input.tensor(0)->info; }

} // namespace neuralnet
