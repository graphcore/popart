#ifndef GUARD_NEURALNET_OP_HPP
#define GUARD_NEURALNET_OP_HPP

#include <boost/optional.hpp>
#include <memory>
#include <set>
#include <poponnx/attributes.hpp>
#include <poponnx/names.hpp>
#include <poponnx/opidentifier.hpp>
#include <poponnx/region.hpp>
#include <poponnx/tensorinfo.hpp>
#include <poponnx/util.hpp>
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

  // The unique identifier of the Op (will always be set in Op::Op)
  OpId id{-1};

  // The operation type, domain & version
  //   A given operator is identified by a three-tuple: (domain, op_type, and
  //   op_version). This is written as domain.op_type:op_version in prose (e.g.,
  //   com.acme.FastConv:3). Nodes in graphs always refer to operators by their
  //   three-part identifier.
  OperatorIdentifier opid;

  struct Settings {

    Settings(Ir &ir_, const std::string &name_) : ir(ir_), name(name_) {}
    virtual ~Settings()        = default;
    Settings(const Settings &) = default;

    Ir &ir;

    std::string name = "";

    // The virtual graph this op has been assigned to if set
    boost::optional<int64_t> vgraphId;

    // If the output should be recomputed if set
    boost::optional<int64_t> recomputeOutput;

    // This method will attempt the optional attributes (vgraphId,
    // recomputeOutput) depending on whether the attribute has been
    // set in the onnx model.
    virtual void setFromAttributes(const Attributes &attributes);
  };

  Settings settings;

  Settings &getSettings() { return settings; }
  const Settings &getSettings() const { return settings; }

  const boost::optional<int64_t> getVirtualGraphId() const {
    return settings.vgraphId;
  }
  void setVirtualGraphId(const boost::optional<int64_t> value) {
    settings.vgraphId = value;
  }

  const boost::optional<int64_t> getRecomputeOutput() const {
    return settings.recomputeOutput;
  }
  void setRecomputeOutput(const boost::optional<int64_t> value) {
    settings.recomputeOutput = value;
  }

  const std::string &getName() const { return settings.name; }

  Ir &getIr() { return settings.ir; }
  const Ir &getIr() const { return settings.ir; }

public:
  Op(const OperatorIdentifier &_opid, const Op::Settings &settings);

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
  std::string debugName() const;

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

  // Disconnect an input test from the op
  void disconnectInTensor(InIndex, Tensor *tensor);

  // Disconnect all input tensors
  void disconnectAllInputs();

  // Disconnect all output tensors
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

  // What are the variants of this Op (if any) which can
  // modify / alias the inputs at the given indices?
  // This function doesn't check for anchor violations
  // or topological order violations. When there are several,
  // they should be returned in descending order of preference
  virtual std::vector<OperatorIdentifier>
  inplaceVariants(const std::vector<InIndex> &) const;

  virtual std::unique_ptr<Op>
  getInplaceVariant(const OperatorIdentifier &) const;

  // The input Region which this Op modifies (for inplace ops)
  virtual view::Region modifies(InIndex) const;
  // The input Region which this Op uses
  virtual view::Region uses(InIndex) const;
  // The input Region which the output will alias (for inplace and view-changing
  // ops)
  virtual view::Region aliases(InIndex) const;
  // Map used regions of the input to/from the output (we assume the same for
  // modifies, aliases, uses)
  virtual view::RegMap fwdRegMap(InIndex) const;
  virtual view::RegMap bwdRegMap(InIndex) const;

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

  const Shape &inShape(InIndex index) const;
  const Shape &outShape(OutIndex index) const;

  Rank inRank(InIndex index) const;
  Rank outRank(OutIndex index) const;

protected:
  // Virtual method to append the op attributes to the stream. This method
  // should be overridden if the derived class has additional attributes.
  virtual void appendAttributes(std::stringstream &,
                                const std::string &tab) const;

  template <typename T>
  void appendAttribute(std::stringstream &ss,
                       const std::string &tab,
                       const std::string &name,
                       const T &value) const {
    ss << tab << name << ":" << value << '\n';
  }

private:
  void appendIO(std::stringstream &) const;
  virtual void appendMore(std::stringstream &) const {}
};

} // namespace poponnx

#endif
