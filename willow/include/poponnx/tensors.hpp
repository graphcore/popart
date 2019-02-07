#ifndef GUARD_NEURALNET_WILLOWTENSORS_HPP
#define GUARD_NEURALNET_WILLOWTENSORS_HPP

#include <vector>
#include <poponnx/names.hpp>
#include <poponnx/vectorandset.hpp>

namespace poponnx {

class Tensors {
public:
  Tensors(Ir &pg);
  ~Tensors() = default;

  Tensor *get(TensorId) const;
  void remove(TensorId);
  bool contains(TensorId) const;

  // create a Variable Tensor
  void addVarInit(const TensorId &, const onnx::TensorProto *);

  // create a Constant Tensor
  void addConstInit(const TensorId &, const onnx::TensorProto *);
  void addConstInit(const TensorId &, const TensorInfo &, const void *);

  // create a Tensor of type Stream
  void addStream(TensorId, const TensorInfo &);
  // create a Tensor of type ActGrad (basically any tensor which is
  // the output of an Op)
  void addActGrad(TensorId);
  std::vector<TensorId> getIds(TensorType) const;
  std::vector<TensorId> getAllTensorIds() const;
  std::vector<TensorId> getNoProducerIds() const;
  const onnx::TensorProto *getOnnxInit(TensorId) const;
  void append(std::stringstream &) const;

  const VectorAndSet &getConstIds() const { return constIds; }
  void insertConstId(const std::string &);
  // remove all Tensors which have no producer and no consumers
  void removeIsolated();

  void updateAliases(Op *op);

  // all non-empty alias Chains to "to"
  // returned map M will always have M[to] = "the identity chain"
  //......"from"...."chains"............................"to"
  //       ^         ^                                   ^
  std::map<Tensor *, view::Chains> aliasChainsTo(Tensor *to) const;

  // all non-empty alias Chains from "from"
  // returned map M will always have M[from] = "the identity chain"
  //......"to"......"chains".............................."from"
  //       ^         ^                                     ^
  std::map<Tensor *, view::Chains> aliasChainsFrom(Tensor *from) const;

  view::Chains getChainsFromTo(Tensor *from, Tensor *to) const;

private:
  // Store the Tensors of type Const
  VectorAndSet constIds;

  std::map<TensorId, std::unique_ptr<Tensor>> M;
  // adds to M, but first confirms that TensorId not already in
  void insert(TensorId, std::unique_ptr<Tensor>);

  void addInit(const TensorId &, const onnx::TensorProto *, TensorType);

  Ir &ir;

  // all non-empty Chains
  //      "to"..............."from"...."chains"
  //       ^                  ^         ^
  std::map<Tensor *, std::map<Tensor *, view::Chains>> aliasChainsToKey;

  // the mirror of the above
  std::map<Tensor *, std::map<Tensor *, view::Chains>> aliasChainsFromKey;

  // return M[t], but with guaranteed identity Chains from t
  std::map<Tensor *, view::Chains>
  getAliasChains(const std::map<Tensor *, std::map<Tensor *, view::Chains>> &M,
                 Tensor *t) const;
};

} // namespace poponnx

#endif
