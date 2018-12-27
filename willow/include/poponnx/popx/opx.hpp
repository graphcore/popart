#ifndef GUARD_NEURALNET_POPOP_HPP
#define GUARD_NEURALNET_POPOP_HPP

#include <poplar/DeviceManager.hpp>
#include <poplar/Engine.hpp>
#include <poplar/Graph.hpp>
#include <poplar/IPUModel.hpp>

#include <poponnx/error.hpp>
#include <poponnx/names.hpp>

namespace poponnx {

class TensorInfo;

namespace popx {

class Devicex;

class Opx {

public:
  // need to pass Devicex down to here, easy
  // access to poplar objects
  Opx(Op *, Devicex *);
  virtual ~Opx();

  // create the input poplar::Tensor for input at index
  // default : throw error (not all Opxs can createInput)
  virtual poplar::Tensor createInput(int index) const;
  // default return false
  virtual bool canCreateInput(int index0) const;
  // If this Opx creates a poplar::Tensor at index0 (via createInput),
  // does it create the same poplar::Tensor as if opx1 creates one at
  // index1?. default behaviour : throws error
  virtual bool createsEquiv(int index0, Opx *opx1, int index1) const;
  // To create a poplar::Tensor for input index index0, which
  // poplar::Tensors must already exist?
  virtual std::vector<TensorId> mustExistBeforeCreate(int index0) const;
  // adds poplar::Tensors to devicex_->popTensors,
  // one for each output of op_.
  virtual void grow(poplar::program::Sequence &) const;
  // clone the poplar::Tensor identified by its TensorId, and copy the contents
  // of it.
  poplar::Tensor cloneNcopy(poplar::program::Sequence &, TensorId) const;
  // clone the poplar::Tensor and copy the contents of it.
  poplar::Tensor cloneNcopy(poplar::program::Sequence &,
                            const poplar::Tensor &) const;
  // Return the poplar Tensor identified by its TensorId, numpy broadcasting it
  // up to the given shape. Throws an exception if the identified Tensor doesn't
  // have a compatible shape.
  poplar::Tensor broadcast(const std::vector<int64_t> &, TensorId) const;
  // Return the given poplar Tensor, numpy broadcasting it up to the given
  // shape. Throws an exception if the given Tensor doesn't have a compatible
  // shape.
  poplar::Tensor broadcast(const std::vector<int64_t> &, poplar::Tensor) const;

  // shortcut for dv_p->graph
  poplar::Graph &graph() const;
  // shortcut for dv_p->tensors.get
  const poplar::Tensor &get(TensorId) const;
  // shortcut for dv_p->tensors.insert
  void insert(TensorId, const poplar::Tensor &) const;
  // shortcut for op_p->input.id(int)
  TensorId inId(InIndex) const;
  // shortcut for op_p->output.id(int)
  TensorId outId(OutIndex) const;
  // shortcut for op_p->input.id(int)
  Tensor *inTensor(InIndex) const;
  // shortcut for op_p->output.id(int)
  Tensor *outTensor(OutIndex) const;
  // shortcut for std::to_string(op_p->id)
  std::string idStr() const;
  // shortcut for op_p->input.tensor(int)->info
  const TensorInfo &inInfo(InIndex) const;
  // shortcut for op_p->input.tensor(int)->info.shape()
  const Shape &inShape(InIndex) const;
  // shortcut for op_p->input.tensor(int)->info
  const TensorInfo &outInfo(OutIndex) const;
  // shortcut for op_p->input.tensor(int)->info.shape()
  const Shape &outShape(OutIndex) const;

  // The Op corresponding to this Opx
  Op *op_p;

  // Generic function to cast the op to it derived type
  // FFS : Use it to replace all the getXXX member function with dynamic_cast
  template <class OP> OP &getOp() const {
    OP *d_op = dynamic_cast<OP *>(op_p);
    if (d_op == nullptr) {
      throw error("Failed to cast to op ({}) derived op ({}), type:{} ",
                  typeid(op_p).name(),
                  typeid(d_op).name(),
                  op_p->opid);
    }
    return *d_op;
  }

  // TODO: Reconsider the names of these two verifyOp functions

  // Generic function to test that op is of a given type
  template <class OP> void verifyOp(Op *op, const OperatorIdentifier &opid) {
    // compare domain, type (Relu, etc.) and version
    if (op->opid != opid) {
      throw error("Cannot create opx for {} from {}", opid, op->opid);
    }
  }

  template <class OP> void verifyOp(Op *op) {
    if (!op->isConvertibleTo<OP>()) {
      throw error("Cannot create opx type from {}", op->opid);
    }
  }

  // The Devicex to which this Opx belongs
  Devicex *dv_p;
};

} // namespace popx
} // namespace poponnx

#endif
