// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_POPOP_HPP
#define GUARD_NEURALNET_POPOP_HPP

#include <poplar/DeviceManager.hpp>
#include <poplar/Engine.hpp>
#include <poplar/Graph.hpp>
#include <poplar/IPUModel.hpp>

#include <popart/error.hpp>
#include <popart/names.hpp>
#include <popart/op.hpp>

namespace popart {

class TensorInfo;

namespace popx {

class ICreatorCandidate;
using ICreatorCandidatePtr = std::shared_ptr<ICreatorCandidate>;
struct UnwindEndpoint;
using UnwindEndpointPtr = std::shared_ptr<UnwindEndpoint>;
struct OpxInAndOutIndex;

class Devicex;
class ViewChangers;

enum class InputCreatorType {
  // Opx has a poplar call to a function that can
  // lay out the input tensor on the device
  CanCreate = 0,
  // Cannot create the input tensor, but can
  // allow an Opx downstream in the graph to
  // create it
  CanUnwind,
  // Can create or unwind
  CanCreateOrUnwind,
  // Cannot create tensor, nor can it allow a
  // a downstream Opx to create the tensor
  Deadend,
  // Has a potential creator, but can also allow an Opx downstream in the graph
  // to create it instead.
  CanDelegate
};

class Opx {

public:
  // need to pass Devicex down to here, easy
  // access to poplar objects
  Opx(Op *, Devicex *);
  virtual ~Opx();

  // create the input poplar::Tensor for input at index with name
  // default : throw error (not all Opxs can createInput)
  virtual poplar::Tensor createInput(InIndex index,
                                     const std::string &name) const;
  // default return DEADEND, i.e. unable to create input tensor, and
  // cannot use downstream opxs as candidates to create input
  // tensor
  virtual InputCreatorType getInputCreatorType(InIndex index) const;
  // When an input tensor has multiple creator candidates, we choose
  // the one with highest priority
  double inputCreatorPriority{0.0};
  // If this Opx creates a poplar::Tensor at index0 (via createInput),
  // does it create the same poplar::Tensor as if opx1 creates one at
  // index1?. default behaviour : throws error
  virtual bool createsEquiv(int index0, const Opx *opx1, int index1) const;

  virtual bool canUnwind(InIndex, OutIndex) const;

  // Reverses the layout change to an input tensor for an op that returned
  // CANUNWIND
  virtual poplar::Tensor
  unwindTensorLayout(poplar::Tensor tensor, InIndex, OutIndex) const;
  virtual view::RegMap unwindRegion(InIndex, OutIndex) const;

  // If the created or unwound tensor does not conform with the IR specs,
  // an Opx may supply a view transformation that transforms that tensor into
  // IR specs
  virtual bool hasCreatorViewChangers(InIndex index) const;
  virtual ViewChangers getCreatorViewChangers(InIndex index) const;

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

  // Returns the Devicex to which this Opx belongs
  const Devicex *getDevicex() const;

  // Access dropoutReferenceTensors from Devicex
  // TODO: Improve see, see TT15790
  std::map<uint32_t, poplar::Tensor> &getDropoutReferenceTensors() const;

  // dv_p->getVirtualGraphId(). Defaults to 0 if virtualGraph is not enabled
  int64_t getVirtualGraphId() const;
  // Returns the virtual graph if enabled, else returns the dv_p->graph
  poplar::Graph &graph() const;
  // shortcut for dv_p->tensors.get
  const poplar::Tensor &get(TensorId) const;
  // shortcut for dv_p->tensors.getView
  const poplar::Tensor &getView(TensorId) const;
  // shortcut for dv_p->tensors.insert
  void insert(TensorId, const poplar::Tensor &) const;

  // shortcut for op_p->input.id(int)
  Tensor *inTensor(InIndex) const;
  // shortcut for op_p->output.id(int)
  Tensor *outTensor(OutIndex) const;
  // the debug prefix to use in poplar calls
  std::string debugPrefix() const;
  std::string debugPrefix(const std::string &prefix) const;
  std::string debugPrefix(const std::string &p1, const std::string &p2) const;
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
    // compare domain, type (Relu, etc.), but not version as an op can support
    // multiple versions
    // TODO : Consider passing in a list of support opid's (for each version)
    if (op->opid.domain != opid.domain || op->opid.type != opid.type) {
      throw error("Cannot create opx for {} from {}", opid, op->opid);
    }
  }

  template <class OP>
  void verifyOp(Op *op, std::vector<OperatorIdentifier> opids) {

    for (auto &opid : opids) {
      if (op->opid == opid) {
        return;
      }
    }

    std::ostringstream oss;
    oss << "In Opx::verifyOp, for op " << op->str()
        << ". Failed to verify, as valid opids are : ( ";
    for (auto valid : opids) {
      oss << valid << ", ";
    }
    oss << ").";
    throw error(oss.str());
  }

  template <class OP> void verifyOp(Op *op) {
    if (!op->isConvertibleTo<OP>()) {
      throw error("Cannot create opx type from {}", op->opid);
    }
  }

  bool hasInput(InIndex) const;
  bool hasOutput(OutIndex) const;

  // Return underlying Poplar input tensor
  const poplar::Tensor &getInTensor(InIndex index) const;

  // Return underlying Poplar output tensor
  const poplar::Tensor &getOutTensor(OutIndex index) const;

  // Return input tensor with shape matching IR specifications
  // (aliases getInTensor, but has any respective ViewChangers applied)
  const poplar::Tensor &getInView(InIndex index) const;

  // Return output tensor with shape matching IR specifications
  // (aliases getOutTensor, but has any respective ViewChangers applied)
  const poplar::Tensor &getOutView(OutIndex index) const;

  bool hasInViewChangers(InIndex index) const;
  const ViewChangers &getInViewChangers(InIndex index) const;
  void setOutViewChangers(OutIndex index, const ViewChangers &changers) const;

  void setOutTensor(OutIndex index, const poplar::Tensor &tensor) const;

  // Input & output tensors used by the opx when it is cached
  std::vector<poplar::Tensor> cachedInputs;
  std::vector<poplar::Tensor> *cachedOutputs = nullptr;

  TensorId inId(InIndex index) const;
  TensorId outId(OutIndex index) const;

  // shortcut for dv_p->getConst
  poplar::Tensor getConst(const poplar::Type &type,
                          const std::vector<size_t> &shape,
                          double val,
                          const std::string &name) const;

  poplar::Tensor getScalarVariable(const poplar::Type &type,
                                   const std::string &name) const;

  // The Opx outputs that come from any subgraph and need to be prepared
  // This allows growing the data flows through subgraphs independently, and
  // growing the Opx that calls the subgraph can be deferred until after all
  // data flows through the called subgraph are grown.
  virtual std::vector<std::tuple<TensorId, TensorId, bool>>
  getOutputsToPrepare() const;

protected:
  // The Devicex to which this Opx belongs
  Devicex *dv_p;

private:
  std::string idStr() const;
};

} // namespace popx
} // namespace popart

#endif
