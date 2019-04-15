#ifndef GUARD_NEURALNET_VOLE_HPP
#define GUARD_NEURALNET_VOLE_HPP

#include <poponnx/op.hpp>

namespace poponnx {

// Examples of Ops with receptive fields include
// ConvOp and AveragePoolOp
class HasReceptiveFieldOp : public Op {
public:
  struct Settings : public Op::Settings {

    Settings(Ir &ir_, const std::string &name_, const Scope &scope_)
        : Op::Settings(ir_, name_, scope_) {}

    std::vector<int64_t> pads;
    std::vector<int64_t> strides;
    std::vector<int64_t> dilations;

    void setFromAttributes(const Attributes &attributes) override;
  };

  HasReceptiveFieldOp(const OperatorIdentifier &_opid,
                      const HasReceptiveFieldOp::Settings &settings);

  // C++ rule of 3 for destructor, copy con, assignment op.

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
  // the spatial dimensions of the operator
  //   : kernel size for convolution
  //   : window size for pooling
  std::vector<int64_t> spatialK;
  // the spatial dimensions of the data
  std::vector<int64_t> spatialD;
  DataType outType;

  void setup() override;
  virtual int64_t getNOutChans() const = 0;
  // return the nSpatialDims lower pads (pads left, bottom)
  std::vector<int64_t> lowerPads() const;
  // return the nSpatialDims upper pads (pads right, top)
  std::vector<int64_t> upperPads() const;

  // backend might prefer a different number format.
  // These convenience functions reduce backend boilerplate.
  // Poponnx uses signed ints of strictly defined sizes, internally.
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

  void appendAttributes(OpSerialiserBase &) const override;

  virtual Shape getOutShape() const;

private:
  // set the public vector "spatialK"
  virtual void setSpatialK() = 0;
  // anything else that a sub-class needs to do should go here:
  virtual void setup0() = 0;
};

} // namespace poponnx

#endif
