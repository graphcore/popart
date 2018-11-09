#ifndef GUARD_NEURALNET_LOSS_HPP
#define GUARD_NEURALNET_LOSS_HPP

#include <map>
#include <poponnx/error.hpp>
#include <poponnx/names.hpp>
#include <poponnx/tensorinfo.hpp>
#include <poponnx/vertex.hpp>

namespace willow {

enum class eLoss { NLL, L1 };
std::map<std::string, eLoss> initLossMap();
const std::map<std::string, eLoss> &lossMap();

class Loss {
public:
  virtual ~Loss()    = default;
  Loss(const Loss &) = default;
  Loss &operator=(const Loss &) = delete;
  Loss(const std::vector<TensorId> &input, TensorId output);
  virtual std::vector<TensorId> getStreamTensorNames() const = 0;
  virtual std::unique_ptr<Op> getOp(Ir *) const              = 0;
  const TensorId &input(int i) const;
  int input_size() const;
  // takes in an int arg to conform
  // with Node function (uses same template)
  const TensorId &output(int) const;
  int output_size() const;
  virtual std::string op_type() const         = 0;
  virtual std::unique_ptr<Loss> clone() const = 0;

private:
  // The names of the input tensors, same
  // format as a Node : "" represents no input
  std::vector<TensorId> input_;
  // The name of the output tensor
  TensorId output_;
};

} // namespace willow

#endif
