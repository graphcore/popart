#ifndef GUARD_NEURALNET_TOPK_HPP
#define GUARD_NEURALNET_TOPK_HPP

#include <popart/op/basesort.hpp>

namespace popart {

class TopKOp : public BaseSortOp {
public:
  TopKOp(const OperatorIdentifier &_opid,
         int64_t k,
         int64_t axis,
         const Op::Settings &settings);
  std::unique_ptr<Op> clone() const override;
  void setup() final;
  virtual void connectInTensor(InIndex, TensorId) final;

  int64_t getK() const;
  std::vector<std::unique_ptr<Op>> getGradOps() final;

  void appendOutlineAttributes(OpSerialiserBase &) const final;

  static InIndex getKInIndex() { return 1; }

  // The outputs are:
  // - the sorted input, sliced from 0:K
  static OutIndex getValuesOutIndex() { return 0; }

  // - the starting indices of the sorted input, sliced from 0:K
  static OutIndex getIndicesOutIndex() { return 1; }

private:
  int64_t K;
};

// Similar to Scatter, except it has 2 inputs instead of 3.
// It is basically Scatter, but with the data input changed to
// a tensor of zeros.
//
// TODO: common base class with Scatter T8198
class TopKGradOp : public Op {
public:
  TopKGradOp(const TopKOp &);

  std::unique_ptr<Op> clone() const final;
  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;
  void setup() final;

  // Which axis the top-k is taken on in the forward Op.
  int64_t getAxis() const;
  const TensorInfo &getGradOutInfo() const;

  // The index at which the gradient of the output of the forward Op is received
  static InIndex gradInIndex() { return 0; }

  // The index at which the indices output of the forward Op is received
  static InIndex indicesInIndex() { return 1; }

  static InIndex gradOutIndex() { return 0; }

  void appendOutlineAttributes(OpSerialiserBase &) const override;

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

private:
  int64_t axis;
  TensorInfo gradOutInfo;
};

} // namespace popart

#endif
