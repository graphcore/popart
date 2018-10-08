#ifndef GUARD_NEURALNET_LOSS_HPP
#define GUARD_NEURALNET_LOSS_HPP

#include <map>
#include <willow/names.hpp>
#include <willow/tensorinfo.hpp>
#include <willow/vertex.hpp>

namespace willow {

class Op;
class Graph;

enum class eLoss { NLL, L1 };
std::map<std::string, eLoss> initLossMap();
const std::map<std::string, eLoss> &lossMap();

class Loss {
public:
  virtual ~Loss() = default;
  Loss(const std::string &);
  virtual std::vector<TensorId> getStreamTensorNames() const = 0;
  const std::vector<std::string> &other() const;
  virtual std::unique_ptr<Op> getOp(Graph *) const = 0;
  const std::vector<std::string> &args() const;

  const TensorId &input(int i) const;
  int input_size() const;
  // takes in an int arg to conform
  // with Node function (uses same template)
  const TensorId &output(int) const;
  int output_size() const;
  virtual std::string op_type() const = 0;

protected:
  void confirmSizes(int nIn, int nArgs) const;

private:
  std::vector<std::string> args_;
  // The names of the input tensors, same
  // format as a Node : "" represents no input
  std::vector<TensorId> input_;
  // The name of the output tensor
  TensorId output_;
};

} // namespace willow

#endif
