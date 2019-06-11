#ifndef GUARD_NEURALNET_LRN_HPP
#define GUARD_NEURALNET_LRN_HPP

#include <poponnx/op.hpp>

namespace poponnx {

class LRNOp : public Op {
public:
  LRNOp(const OperatorIdentifier &_opid,
        float _alpha,
        float _beta,
        float _bias,
        int64_t _size,
        const Op::Settings &settings_);

  std::unique_ptr<Op> clone() const final;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
  void setup() final;

  // Inputs
  static InIndex getInIndex() { return 0; }

  // Outputs
  static OutIndex getOutIndex() { return 0; }

  float getSubgraphValue() const final { return getHighSubgraphValue(); }

  // Attributes
  float getAlpha() const { return alpha; }
  float getBeta() const { return beta; }
  float getBias() const { return bias; }
  int64_t getSize() const { return size; }

  void appendAttributes(OpSerialiserBase &) const override;

private:
  float alpha;
  float beta;
  float bias;
  int64_t size;
};

class LRNGradOp : public Op {
public:
  LRNGradOp(const LRNOp &);
  std::unique_ptr<Op> clone() const final;
  void setup() final;

  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;

  // Attributes
  float getAlpha() const { return alpha; }
  float getBeta() const { return beta; }
  float getBias() const { return bias; }
  int64_t getSize() const { return size; }

  void appendAttributes(OpSerialiserBase &) const override;

  // Inputs
  static InIndex getInIndex() { return 0; }
  static InIndex getFwdInInIndex() { return 1; }

  // Outputs
  static OutIndex getOutIndex() { return 0; }

  float getSubgraphValue() const final { return getHighSubgraphValue(); }

private:
  float alpha;
  float beta;
  float bias;
  int64_t size;
};

} // namespace poponnx

#endif
