#ifndef GUARD_NEURALNET_OP_HPP
#define GUARD_NEURALNET_OP_HPP

#include <memory>
#include <set>
#include <poponnx/attributes.hpp>
#include <poponnx/names.hpp>
#include <poponnx/opidentifier.hpp>
#include <poponnx/tensorinfo.hpp>
#include <poponnx/vertex.hpp>

namespace poponnx {

// the input tensor of a grad-op has what kind of
// relationship with the corresponding non-grad-op?
// design note: it's not possible for an input to a
// grad-op to NOT be directly related to
// the corresponding non-grad-op.
enum class GradOpInType { IN = 0, OUT, GRADOUT };

class GradInOutMapper {
public:
  GradInOutMapper(InIndex iGrad_, int iNonGrad_, GradOpInType);
  // input index to a grad-op
  InIndex iGrad;
  // "input/output/gradient-of-output" index to
  // corresponding non-grad op,
  int iNonGrad;
  // where "input/output/gradient-of-output" above is
  GradOpInType type;

  bool operator==(const GradInOutMapper &rhs) const;
};

class Op : public Vertex {
public:
  // We use pointers to TensorIndexMaps for PIMPL reasons.
  // Note that we cannot initialise these with {nullptr} on gcc.
  // They are initialised in the Op constuctors
  // The consumed Tensors
  std::unique_ptr<TensorIndexMap> input;
  // The produced Tensors
  std::unique_ptr<TensorIndexMap> output;

  // all Ops will be topologically sorted "as close to" the order of
  // priority (highest to lowest) while still resulting in a valid
  // topological ordering.
  // default : 0.0
  double priority{0.0};

  // the Ir to which the Op belongs
  Ir *pir;

  // The unique identifier of the Op (will always be set in Op::Op)
  OpId id{-1};

  // The operation type, domain & version
  //   A given operator is identified by a three-tuple: (domain, op_type, and
  //   op_version). This is written as domain.op_type:op_version in prose (e.g.,
  //   com.acme.FastConv:3). Nodes in graphs always refer to operators by their
  //   three-part identifier.
  OperatorIdentifier opid;

  // attributes from the Node, if it was created from ONNX
  const Attributes nAtts;

public:
  Op(const OperatorIdentifier &_opid,
     Ir *_ir,
     const std::string &name = {},
     const Attributes &_attr = {});
  // Note: copy constructor does NOT copy input and output
  Op(const Op &);
  Op &operator=(const Op &) = delete;
  // A c++ aside: the rule-of-3 says that it's good
  // practise to have an explicit destructor,
  // given that there is an explict copy con.
  // But not really nec. as Vertex has a virtual
  // destructor.
  virtual ~Op();

  std::string str() const final;

  // create an ActGrad (output) tensor
  // and wire it to this Op's output
  void createAndConnectOutTensor(OutIndex, TensorId);

  void append(std::stringstream &ss) const;

  // sum of the total memory of all output tensors
  // We might want a cycle counter too for more sophisticated recomputation
  int64_t memOfOutputs() const;

  // wire a tensor to input: updates input and
  // updates consumers of tensor with id TensorId
  void defaultConnectInTensor(InIndex, TensorId);

  virtual void connectInTensor(InIndex, TensorId);

  void connectOutTensor(OutIndex, TensorId);

  void disconnectAllInputs();
  void disconnectAllOutputs();

  // might the input tensors be modified?
  bool mayModify(InIndex) const;

  const std::string &name() const;

  // set shape and type parameters,
  // This function MUST set output
  // TensorInfos for all outputs
  virtual void setup();

  // return a vector of 1 or several gradient Ops: for
  // obtaining the gradient of the inputs of this Op.
  // If this Op is already a gradient Op, throws error
  // Why is this not constant? For one, nOps counter increments.
  virtual std::vector<std::unique_ptr<Op>> getGradOps();

  // Can the input at inIndex be modified inplace to
  // become the output at index 0?
  // This function doesn't check for anchor violations
  // or topological order violations
  virtual bool hasInplaceVariant(InIndex) const;

  // get the inplace Op described above
  virtual std::unique_ptr<Op> getInplaceVariant(InIndex);

  // Does this Op modify the input at index InIndex? Default: No.
  virtual bool modifies(InIndex) const;

  // A grad-op outputs an edge-gradient tensor dT at gradOpOutIndex.
  // dT is the edge-gradient of a tensor T which was the input
  // to grad-op's non-grad partner. At what index was T the input
  // of non-grad-op? If not relevant (non-grad-ops) throw an error
  virtual int getNonGradInIndex(int gradOpOutIndex) const;

  // For grad-ops, matching input indices to
  // corresponding IN/OUT/GRADOUT indices of
  // corresponding non-grad-op.
  // throws an error if not appropriate (non-grad ops)
  virtual const std::vector<GradInOutMapper> &gradInputInfo() const;

  // return the full map corresponding to getNonGradInIndex.
  // throws an error if not appropriate (non-grad)
  virtual const std::map<int, int> &gradOutToNonGradIn() const;

  // for non-grad-op `op', takes in the set of output indices
  // of `op' for which a gradient is available and returns
  // if all the gradients needed to create grad-ops are present
  // currently this will just compare the size of
  // the set passed in with number of paths to final loss
  bool readyToCreateGradients(std::set<int> &) const;

  // return a copy of self, similar to
  // cpppatterns.com/patterns/virtual-constructor.html
  // some people call it "covariant return type"
  // Throws error from this class if not implemented
  virtual std::unique_ptr<Op> clone() const;

  template <typename T> bool isConvertibleTo() const {
    return dynamic_cast<const T *>(this) != nullptr;
  }

  // Is this Op a LossOp (nll, l1loss, etc)? Note:
  // the Sum op which adds the losses together is not
  // a LossOp (although its Phase is LOSS)
  virtual bool isLossOp() const;

  // helper functions, access fields of input and output
  Tensor *inTensor(InIndex index);
  const Tensor *inTensor(InIndex index) const;
  Tensor *outTensor(OutIndex index);
  const Tensor *outTensor(OutIndex index) const;

  TensorInfo &inInfo(InIndex index);
  const TensorInfo &inInfo(InIndex index) const;
  TensorInfo &outInfo(OutIndex index);
  const TensorInfo &outInfo(OutIndex index) const;

  const Shape &inShape(InIndex index);
  const Shape &outShape(OutIndex index);

  Rank inRank(InIndex index);
  Rank outRank(OutIndex index);

private:
  void appendIO(std::stringstream &) const;
  virtual void appendMore(std::stringstream &) const {}

  // A user supplied name
  std::string _name;
};

} // namespace poponnx

#endif
