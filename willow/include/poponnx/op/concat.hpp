#ifndef GUARD_NEURALNET_CONCAT_HPP
#define GUARD_NEURALNET_CONCAT_HPP

#include <poponnx/op.hpp>

namespace poponnx {

class ConcatOp : public Op {
public:
  ConcatOp(const OperatorIdentifier &_opid,
           int64_t axis_,
           const Op::Settings &settings);

  std::unique_ptr<Op> clone() const override;
  void setup() override;

  std::vector<std::unique_ptr<Op>> getGradOps() final;

  int64_t getAxis() const;

  // note that this is not final, ConcatInplaceOp overrides it
  std::vector<OperatorIdentifier>
  inplaceVariants(const std::vector<InIndex> &) const override;

  std::unique_ptr<Op> getInplaceVariant(const OperatorIdentifier &,
                                        const std::vector<InIndex> &) override;

  static InIndex getInIndex(InIndex index) { return index; }
  static OutIndex getOutIndex() { return 0; }

  void appendAttributes(std::stringstream &ss,
                        const std::string &tab) const override;

private:
  int64_t axis = 0;
};

// An inplace variant of the concat op
class ConcatInplaceOp : public ConcatOp {
public:
  ConcatInplaceOp(const ConcatOp &concatOp, int64_t axis_);

  std::unique_ptr<Op> clone() const override;

  std::vector<OperatorIdentifier>
  inplaceVariants(const std::vector<InIndex> &) const final {
    return {};
  }

  std::unique_ptr<Op> getInplaceVariant(const OperatorIdentifier &o,
                                        const std::vector<InIndex> &i) final {
    // this throws an error
    return Op::getInplaceVariant(o, i);
  }

  std::unique_ptr<RegionIOMap>
  aliases(const std::map<InIndex, Shape> &) const final;
};

class ConcatGradOp : public Op {
public:
  ConcatGradOp(const ConcatOp &op, InIndex input);
  ConcatGradOp(const ConcatInplaceOp &op, InIndex input);

  std::unique_ptr<Op> clone() const override;
  void setup() override;

  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;

  int64_t getAxis() const;
  int64_t getStart() const;
  int64_t getEnd() const;

  static InIndex getInIndex() { return 0; }
  static OutIndex getOutIndex() { return 0; }

protected:
  // An unsafe constructor that allows using any OperatorIdentifier
  ConcatGradOp(const OperatorIdentifier &_opid,
               const ConcatGradOp &concat_grad_op);

private:
  int64_t axis;
  int64_t start;
  int64_t end;
  InIndex fwdInput;

  TensorInfo gradInfo;
  std::map<int, int> gradOutToNonGradInInfo;
};

} // namespace poponnx

#endif
