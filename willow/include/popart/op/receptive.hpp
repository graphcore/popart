// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_VOLE_HPP
#define GUARD_NEURALNET_VOLE_HPP

#include <popart/op.hpp>

namespace popart {

enum class AutoPad { NOTSET = 0, SAME_UPPER, SAME_LOWER, VALID };

// Examples of Ops with receptive fields include
// MaxPoolOp and AveragePoolOp
class HasReceptiveFieldOp : public Op {
public:
  struct ReceptiveOpAttributes {
    std::vector<int64_t> pads;
    std::vector<int64_t> outPads;
    std::vector<int64_t> strides;
    std::vector<int64_t> dilations;
    std::vector<int64_t> inDilations;
    std::string auto_pad;
    int64_t ceil_mode = 0;

    void setFromAttributes(const Attributes &attributes);
  };

  HasReceptiveFieldOp(
      const OperatorIdentifier &_opid,
      const HasReceptiveFieldOp::ReceptiveOpAttributes &attributes,
      const Op::Settings &settings);
  std::unique_ptr<Op> clone() const override = 0;

  int getNSpatialDims() const;
  int64_t getBatchSize() const;
  int64_t getNInChans() const;

  virtual Shape getSpatialK() const = 0;
  Shape getStrides() const;
  Shape getLowerPads() const { return lowerPads(); }
  Shape getUpperPads() const { return upperPads(); }
  Shape getLowerOutPads() const { return lowerOutPads(); }
  Shape getUpperOutPads() const { return upperOutPads(); }
  Shape getPads() const;
  Shape getOutPads() const;
  Shape getDilations() const;
  Shape getInDilations() const;

  const std::vector<int64_t> basePads;
  const std::vector<int64_t> baseOutPads;
  const std::vector<int64_t> baseStrides;
  const std::vector<int64_t> baseDilations;
  const std::vector<int64_t> baseInDilations;

  const AutoPad padType;
  const bool ceilMode;

  static AutoPad getAutoPad(const std::string &autoPadStr);
  std::string getAutoPadStr(const AutoPad &x) const;

  static void alterPads(Shape &pads_,
                        Shape OutShape_,
                        Shape spatialD_,
                        Shape spatialK_,
                        std::vector<int64_t> strides_);

  // the spatial dimensions of the data
  std::vector<int64_t> getSpatialD() const;
  // the spatial dimensions of the output
  std::vector<int64_t> getSpatialO() const;

  void setup() override;
  virtual int64_t getNOutChans() const = 0;
  // return the nSpatialDims lower pads (pads left, bottom)
  std::vector<int64_t> lowerPads() const;
  static std::vector<int64_t>
  lowerPads(Shape pads, int nSpatialDims, AutoPad padType);

  // return the nSpatialDims upper pads (pads right, top)
  std::vector<int64_t> upperPads() const;
  static std::vector<int64_t>
  upperPads(Shape pads, int nSpatialDims, AutoPad padType);

  std::vector<int64_t> lowerOutPads() const;
  std::vector<int64_t> upperOutPads() const;

  // backend might prefer a different number format.
  // These convenience functions reduce backend boilerplate.
  // Popart uses signed ints of strictly defined sizes, internally.
  std::vector<size_t> spatialD_szt() const;
  std::vector<size_t> spatialK_szt() const;
  std::vector<uint32_t> lowerPads_u32() const;
  std::vector<uint32_t> upperPads_u32() const;
  std::vector<int> lowerPads_i32() const;
  std::vector<int> upperPads_i32() const;
  std::vector<uint32_t> dilations_u32() const;
  std::vector<uint32_t> strides_u32() const;

  static InIndex getInIndex() { return 0; }
  static OutIndex getOutIndex() { return 0; }

  void appendOutlineAttributes(OpSerialiserBase &) const override;

  // Determine the spatial dimensions of the output - a subset of the
  // dimensions of the complete output shape
  static Shape getSpatialOutShape(Shape spatialD_,
                                  Shape spatialK_,
                                  std::vector<int64_t> pads_,
                                  std::vector<int64_t> outPads_,
                                  std::vector<int64_t> strides_,
                                  std::vector<int64_t> dilations_,
                                  std::vector<int64_t> inDilations_,
                                  AutoPad auto_pad_,
                                  bool ceil_mode_ = false);

private:
  Shape getOutShape(const Shape &pads) const;

  // anything else that a sub-class needs to do should go here:
  virtual void setup0() const = 0;
};

} // namespace popart

#endif
