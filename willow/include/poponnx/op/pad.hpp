#ifndef GUARD_NEURALNET_PAD_HPP
#define GUARD_NEURALNET_PAD_HPP

#include <poponnx/op.hpp>

namespace poponnx {

class PadOp : public Op {
public:
  PadOp(const OperatorIdentifier &_opid,
        const std::vector<int64_t> &_pads,
        float value_,
        const std::string &_mode,
        const Op::Settings &settings_);

  std::unique_ptr<Op> clone() const final;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
  // returns true of all pad size in all dimensions
  // and on both sides, are zero
  bool padSizeZero() const;
  void setup() final;

  static InIndex getInIndex() { return 0; }
  static OutIndex getOutIndex() { return 0; }

  // The region of the output tensors which is based on the input tensor value.
  // The complement of this region is the padding region.
  view::Region valueRegion() const;
  std::vector<int64_t> padDimensions() const;

  const std::vector<int64_t> &getPads() const;
  float getPadValue() const;
  const std::string &getMode() const;

  void appendAttributes(OpSerialiserBase &) const override;

  bool canBeReplacedByIdentity() override;

  float getSubgraphValue() const final { return 0.1f; }

private:
  std::vector<int64_t> pads;
  float pad_value;
  std::string mode;
};
} // namespace poponnx

#endif
