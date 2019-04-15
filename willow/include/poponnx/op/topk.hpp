#ifndef GUARD_NEURALNET_TOPK_HPP
#define GUARD_NEURALNET_TOPK_HPP

#include <poponnx/op/basesort.hpp>

namespace poponnx {

// Opset version 1 from
// https://github.com/onnx/onnx/blob/master/docs/Operators.md#TopK
// (The definition of Top-K changes in opset 10)

class TopKOp : public BaseSortOp {
public:
  TopKOp(const OperatorIdentifier &_opid,
         int64_t k,
         int64_t axis,
         const Op::Settings &settings);
  std::unique_ptr<Op> clone() const override;
  void setup() final;

  int64_t getK() const;

  void appendAttributes(OpSerialiserBase &) const final;

  // The outputs are:
  // - the sorted input, sliced from 0:K
  static OutIndex getValuesOutIndex() { return 0; }

  // - the starting indices of the sorted input, sliced from 0:K
  static OutIndex getIndicesOutIndex() { return 1; }

private:
  const int64_t K;
};

} // namespace poponnx

#endif
