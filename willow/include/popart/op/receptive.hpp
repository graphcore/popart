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
  struct Settings : public Op::Settings {

    Settings(Graph &graph_, const std::string &name_, const Scope &scope_)
        : Op::Settings(graph_, name_, scope_) {}

    std::vector<int64_t> pads;
    std::vector<int64_t> strides;
    std::vector<int64_t> dilations;
    std::string auto_pad;

    void setFromAttributes(const Attributes &attributes) override;
  };

  HasReceptiveFieldOp(const OperatorIdentifier &_opid,
                      const HasReceptiveFieldOp::Settings &settings);

  int nSpatialDims;
  int64_t batchSize;
  int64_t nInChans;

  Shape getSpatialK() const { return spatialK; }
  Shape getStrides() const { return strides; }
  Shape getLowerPads() const { return lowerPads(); }
  Shape getUpperPads() const { return upperPads(); }

  std::vector<int64_t> pads;
  std::vector<int64_t> strides;
  std::vector<int64_t> dilations;

  AutoPad padType;

  static AutoPad getAutoPad(const std::string &autoPadStr);
  std::string getAutoPadStr(const AutoPad &x) const;

  static void alterPads(Shape &pads_,
                        Shape OutShape_,
                        Shape spatialD_,
                        Shape spatialK_,
                        std::vector<int64_t> strides_);

  // the spatial dimensions of the operator
  //   : kernel size for convolution
  //   : window size for pooling
  std::vector<int64_t> spatialK;
  // the spatial dimensions of the data
  std::vector<int64_t> spatialD;
  DataType outType;
  // the spatial dimensions of the output
  std::vector<int64_t> spatialO;

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

  virtual Shape getOutShape() const;
  // Determine the spatial dimensions of the output - a subset of the
  // dimensions of the complete output shape
  static Shape getSpatialOutShape(Shape spatialD_,
                                  Shape spatialK_,
                                  std::vector<int64_t> pads_,
                                  std::vector<int64_t> strides_,
                                  std::vector<int64_t> dilations_,
                                  AutoPad auto_pad_);

private:
  // set the public vector "spatialK"
  virtual void setSpatialK() = 0;
  // anything else that a sub-class needs to do should go here:
  virtual void setup0() = 0;
};

} // namespace popart

#endif
