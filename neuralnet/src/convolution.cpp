// TIMELINE:
// 1) support basic conv nets.

#pragma clang diagnostic push // start ignoring warnings
#pragma clang diagnostic ignored "-Weverything"
#include <cblas.h>
#include <onnx/onnx.pb.h>
#pragma clang diagnostic pop // stop ignoring warnings
#include <map>

namespace neuralnet {

// Tensors to log every iteration
// Also, frequency at which to return all weights
// TODO(jn) ask David Norman how tensorflow does this.
class Recorder {};

// Learning scheduler
// momentum, learning rates, etc.
class Schedule {};

class Op;
class Tensor;

class Graph {
public:
  Graph(onnx::ModelProto &&,
        Recorder &&,
        // Schedule needed, if momentum the graph is different
        Schedule &&,
        // Weights tensors which are not to be updated
        const std::vector<std::string> &constTensors);

  onnx::ModelProto step(int n);

private:
  const onnx::ModelProto onnxModel;
  Recorder recorder;
  Schedule schedule;
  std::map<Tensor *, std::unique_ptr<Tensor>> tensors;
  std::map<Op *, std::unique_ptr<Op>> ops;
};


enum class TensorType {
  Activation = 0,
  Const,
  Gradient,
  Momentum,
  Other,
  Stream,
  Unknown,
  Variable
};

class Tensor {
public:
private:
  TensorType type;
  Op *producer;
  std::vector<Op *> consumers;
  std::vector<int64_t> shape;
};

class Op {
  std::vector<Tensor *> ins;
  std::vector<Tensor *> outs;
  std::map<Tensor *, int> inIndices;
  // might the input tensors be modified?
  std::map<Tensor *, bool> isConst;
  double priority;
};

Graph::Graph(onnx::ModelProto &&inMod,
             Recorder &&rec,
             // Schedule needed, if momentum the graph is different
             Schedule &&sched,
             // Weights tensors which are not to be updated
             const std::vector<std::string> &constTensors)
    : onnxModel(inMod), recorder(rec), schedule(sched) {
  auto &onnxGraph = onnxModel.graph();
  auto &onnxNodes = onnxGraph.node();
  for (const auto &node : onnxNodes) {
    // strip off the message fields, pass them into this constructor
    std::unique_ptr<Op> op (new Op());
    ops[op.get()] = std::move(op);

  }
}

int convolution() { return 0; }

} // namespace neuralnet
