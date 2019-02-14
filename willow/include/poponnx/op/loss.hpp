#ifndef GUARD_NEURALNET_LOSS_HPP
#define GUARD_NEURALNET_LOSS_HPP

#include <boost/optional.hpp>
#include <map>
#include <poponnx/error.hpp>
#include <poponnx/names.hpp>
#include <poponnx/op.hpp>
#include <poponnx/tensorinfo.hpp>

namespace poponnx {

enum class eLoss { NLL, L1 };
std::map<std::string, eLoss> initLossMap();
const std::map<std::string, eLoss> &lossMap();

class Loss {
public:
  virtual ~Loss()    = default;
  Loss(const Loss &) = default;
  Loss &operator=(const Loss &) = delete;
  Loss(const std::vector<TensorId> &input, TensorId output);
  virtual std::vector<TensorId> getStreamTensorNames() const             = 0;
  virtual std::unique_ptr<Op> getOp(const Op::Settings &settings_) const = 0;
  const TensorId &input(InIndex i) const;
  int input_size() const;
  // takes in an int arg to conform
  // with Node function (uses same template)
  const TensorId &output(OutIndex) const;
  int output_size() const;
  virtual const OperatorIdentifier &op_type() const = 0;
  virtual std::unique_ptr<Loss> clone() const       = 0;

  void virtualGraph(int64_t value) { vgraphId = value; }

protected:
  // Identify on which vgraph the loss should be executed. This is
  // an optional setting and may not be valid when virtual graph's are not
  // enabled
  boost::optional<int64_t> vgraphId;

private:
  // The names of the input tensors, same
  // format as a Node : "" represents no input
  std::vector<TensorId> input_;
  // The name of the output tensor
  TensorId output_;
};

class LossOp : public Op {
public:
  LossOp(const OperatorIdentifier &_opid, const Op::Settings &settings_);
  LossOp(const Op &);

  bool isLossOp() const override;
};

} // namespace poponnx

#endif
