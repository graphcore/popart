

#include <poponnx/op.hpp>
#include <poponnx/op/subgraph.hpp>
#include <poponnx/popx/devicex.hpp>
#include <poponnx/popx/opx.hpp>
#include <poponnx/popx/opxmanager.hpp>

#include <poputil/GraphFunction.hpp>

namespace poponnx {
namespace popx {

static inline poputil::graphfn::Signature
extendWithOutputs(poputil::graphfn::Signature s, unsigned count) {
  std::fill_n(std::back_inserter(s), count, poputil::graphfn::created());
  return s;
}

// The SubgraphTensorFunction that takes n inputs and produces m outputs
class SubgraphTensorFunction {
  poputil::graphfn::VoidFunction voidFunc;
  unsigned numOutputs;

public:
  SubgraphTensorFunction(
      poplar::Graph &graph,
      poputil::graphfn::Signature sig,
      unsigned numOutputs_,
      std::function<std::vector<poplar::Tensor>(std::vector<poplar::Tensor> &,
                                                poplar::program::Sequence &)> f)
      : voidFunc(graph,
                 extendWithOutputs(std::move(sig), numOutputs_),
                 [&](std::vector<poplar::Tensor> &args,
                     poplar::program::Sequence &seq) {
                   auto numInputs = args.size() - numOutputs_;
                   auto outputs   = f(args, seq);
                   std::copy(outputs.begin(),
                             outputs.end(),
                             args.begin() + numInputs);
                 }),
        numOutputs(numOutputs_) {}

  std::vector<poplar::Tensor> operator()(std::vector<poplar::Tensor> &args,
                                         poplar::program::Sequence &seq) {
    // Append the output tensors
    std::fill_n(std::back_inserter(args), numOutputs, poplar::Tensor());

    // Call the tensor function
    voidFunc(args, seq);

    // Create a vector of the output tensors
    auto numInputs = args.size() - numOutputs;
    std::vector<poplar::Tensor> t(args.begin() + numInputs, args.end());

    // Revert the input to the original state
    args.resize(args.size() - numOutputs);
    return t;
  }
};

class SubgraphTensorFunctionCache {
public:
  // Return a map of the subgraph caches
  static std::map<int64_t, SubgraphTensorFunction> &getCache() {
    static std::map<int64_t, SubgraphTensorFunction> subgraphCache;
    return subgraphCache;
  }

  static void add(int64_t key, SubgraphTensorFunction &function) {
    auto &cache = getCache();
    if (cache.find(key) == cache.end()) {
      cache.emplace(key, function);
    } else {
      throw error(
          "TensorFunction key {} already exisiting in tensor function cache",
          key);
    }
  }

  // remove the tensor function for this key if it exists
  static void remove(int64_t key) { getCache().erase(key); }

  static boost::optional<SubgraphTensorFunction> find(int64_t key) {
    auto &cache = getCache();
    auto it     = cache.find(key);
    if (it != cache.end()) {
      return it->second;
    } else {
      return boost::none;
    }
  }
};

class SubgraphOpx : public Opx {

  // Need to store the cache id so we can use it in the destructor to
  // deregister the tensorfunction
  int64_t subgraphId;

  // List of opx's created for this subgraph
  std::vector<Opx *> opxs;

public:
  SubgraphOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
    SubgraphOp &sgop = getOp<SubgraphOp>();
    subgraphId       = sgop.getSubgraphId();

    // Create the opx's for the child ops of the subgraph
    for (auto &_op : sgop.getChildOpsInfo()) {
      std::unique_ptr<Opx> opx = dv_p->createOpx(_op.op);
      opxs.push_back(opx.get());
      devicex->opxs.insert({_op.op->id, std::move(opx)});
    }
  }

  virtual ~SubgraphOpx() { SubgraphTensorFunctionCache::remove(subgraphId); }

  bool isInputModified(const Op *op, InIndex index) const {
    // The input is modified or alised then we consider it modified.
    return (op->aliases(index).isEmpty() == false ||
            op->modifies(index).isEmpty() == false);
  }

  void grow(poplar::program::Sequence &sequence) const {
    SubgraphOp &sgop = getOp<SubgraphOp>();

    std::vector<poplar::Tensor> outputs;
    std::vector<poplar::Tensor> inputs;

    // Create a list of inputs across all ops in the subgraph
    {
      int inputIndex = 0;
      for (auto &childOp : sgop.getChildOpsInfo()) {
        for (auto &input : childOp.inputs) {
          if (input.second.external) {
            if (cachedInputs.empty()) {
              inputs.push_back(dv_p->tensors.get(input.second.id));
            } else {
              // This is a subgraph in a subgraph, all inputs should be in the
              // cached list
              inputs.push_back(cachedInputs[inputIndex++]);
            }
          }
        }
      }
    }

    // Find the subgraph tensor function in the cache
    auto tensorFunction = SubgraphTensorFunctionCache::find(subgraphId);
    if (tensorFunction) {
      // Call the exisiting tensor function
      logging::devicex::info("Reusing subgraph tensor function for key={}",
                             subgraphId);
      outputs = (*tensorFunction)(inputs, sequence);
    } else {
      // Create a new tensor function
      logging::devicex::info("Creating new subgraph tensor function for key={}",
                             subgraphId);

      unsigned numInputs  = 0;
      unsigned numOutputs = 0;

      int inputIndex = 0;

      // Build the signature of the tensor function
      poputil::graphfn::Signature sig;
      for (auto &childOp : sgop.getChildOpsInfo()) {

        for (auto &input : childOp.inputs) {
          if (input.second.external) {

            // Is the input tensor modified by the op, inwhich case we need to
            // use the graphfn::inout signature
            bool inputModified = isInputModified(childOp.op, input.first);

            if (cachedInputs.empty()) {
              if (inputModified) {
                sig.push_back(poputil::graphfn::inout(
                    get(input.second.id), input.second.id + "_inout"));
              } else {
                sig.push_back(poputil::graphfn::input(get(input.second.id),
                                                      input.second.id + "_in"));
              }
            } else {
              // This is a subgraph in a subgraph, all inputs should be in the
              // cached list
              if (inputModified) {
                sig.push_back(poputil::graphfn::inout(
                    cachedInputs[inputIndex++], input.second.id + "_inout"));
              } else {
                sig.push_back(poputil::graphfn::input(
                    cachedInputs[inputIndex++], input.second.id + "_in"));
              }
            }

            numInputs++;
          }
        }

        for (auto &output : childOp.outputs) {
          if (output.second.external) {
            numOutputs++;
          }
        }
      }

      auto function = SubgraphTensorFunction(
          graph(),
          sig,
          numOutputs,
          [&](std::vector<poplar::Tensor> &args,
              poplar::program::Sequence &prog) -> std::vector<poplar::Tensor> {
            int subGraphInputIndex = 0;

            std::map<TensorId, poplar::Tensor> internalTensors;
            std::vector<poplar::Tensor> tensorFunctionOutputs;

            int childOpIndex = 0;

            for (auto &opx : opxs) {

              // fill in the inputs
              logging::devicex::debug("Subgraph key={} growing op {}",
                                      sgop.getSubgraphId(),
                                      opx->op_p->str());

              // Add the input for the child op to the cachedInputs
              auto &childOpInfo = sgop.getChildOpsInfo()[childOpIndex++];
              for (auto &t : childOpInfo.inputs) {
                if (t.second.external) {
                  opx->cachedInputs.push_back(args[subGraphInputIndex++]);
                } else {
                  opx->cachedInputs.push_back(internalTensors[t.second.id]);
                }
              }

              // Setup the cached outputs for the op
              std::vector<poplar::Tensor> opxOutputs;
              opx->cachedOutputs = &opxOutputs;

              // grow the opx - will make calls to poplar to add to the sequence
              opx->grow(prog);

              // Save the outputs for the opx
              for (int outputIndex = 0; outputIndex < opxOutputs.size();
                   outputIndex++) {
                auto &tensor     = opxOutputs[outputIndex];
                auto &tensorInfo = childOpInfo.outputs[outputIndex];

                internalTensors.insert({tensorInfo.id, tensor});

                // If the output is external to the subgraph out it in the
                // tensorFunctionOutputs list
                if (tensorInfo.external)
                  tensorFunctionOutputs.push_back(tensor);
              }
            }

            // Return all the outputs
            return tensorFunctionOutputs;
          });

      // Add the tensor function the cache
      SubgraphTensorFunctionCache::add(sgop.getSubgraphId(), function);

      // Call the tensor function
      outputs = function(inputs, sequence);
    }

    // Store the outputs
    for (int i = 0; i < outputs.size(); ++i) {
      setOutTensor(i, outputs[i]);
    }
  }
};

namespace {
OpxCreator<SubgraphOpx> subgraphOpxCreator(Onnx::CustomOperators::Subgraph);
}

} // namespace popx
} // namespace poponnx
