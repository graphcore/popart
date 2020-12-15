// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_WILLOWTENSORS_HPP
#define GUARD_NEURALNET_WILLOWTENSORS_HPP

#include <unordered_map>
#include <vector>
#include <popart/aliases.hpp>
#include <popart/chains.hpp>
#include <popart/names.hpp>
#include <popart/vectorandset.hpp>

namespace popart {

class Tensors {
public:
  Tensors(Graph &pg);
  ~Tensors() = default;

  Tensor *get(TensorId) const;
  void remove(TensorId);
  bool contains(TensorId) const;

  // Search for a tensor with a scope
  // Return the scoped tensorId
  TensorId find(TensorId, const Scope &) const;
  // Search for a tensor with a scope
  bool contains(TensorId, const Scope &) const;

  // create a Variable Tensor
  void addVarInit(const TensorId &,
                  const ONNX_NAMESPACE::TensorProto *,
                  const DebugContext &dc = {});
  void addVarInit(const TensorId &,
                  const TensorInfo &,
                  const void *,
                  const DebugContext &dc = {});

  // create a Constant Tensor
  void addConstInit(const TensorId &,
                    const ONNX_NAMESPACE::TensorProto *,
                    const DebugContext &dc = {});
  void addConstInit(const TensorId &,
                    const TensorInfo &,
                    const void *,
                    const DebugContext &dc = {});

  // make an existing tensor a const init tensor
  void makeConstInit(const TensorId &, const void *);

  // create a Tensor of type Stream
  void addStream(TensorId, const TensorInfo &, const DebugContext &dc = {});
  // create a Tensor of type ActGrad (basically any tensor which is
  // the output of an Opm)
  void addActGrad(TensorId, const DebugContext &dc = {});
  std::vector<TensorId> getIds(TensorType) const;
  std::vector<Tensor *> getOfType(TensorType) const;
  std::vector<TensorId> getAllTensorIds() const;
  std::vector<TensorId> getNoProducerIds() const;
  void append(std::stringstream &) const;

  const VectorAndSet &getConstIds() const { return constIds; }
  void insertConstId(const std::string &);
  // remove all Tensors which have no producer and no consumers
  void removeIsolated(bool retainRemote);

  const Aliases &getAliases() const { return aliases; }
  void clearAliases();
  void updateAliases(Op *op);
  view::Regions getAliasRegions(Tensor *from, Tensor *to) const;

  // all non-empty alias Chains to "to"
  // returned map M will always have M[to] = "the identity chain"
  //......"from"...."chains"............................"to"
  //       ^         ^                                   ^
  std::unordered_map<Tensor *, view::Chains> aliasChainsTo(Tensor *to) const;

  // all non-empty alias Chains from "from"
  // returned map M will always have M[from] = "the identity chain"
  //......"to"......"chains".............................."from"
  //       ^         ^                                     ^
  std::unordered_map<Tensor *, view::Chains>
  aliasChainsFrom(Tensor *from) const;

  view::Chains getChainsFromTo(Tensor *from, Tensor *to) const;

  TensorId moveIntoTensors(std::unique_ptr<Tensor> tensor);

private:
  // Store the Tensors of type Const
  VectorAndSet constIds;

  std::unordered_map<TensorId, std::unique_ptr<Tensor>> M;
  // adds to M, but first confirms that TensorId not already in
  void insert(TensorId, std::unique_ptr<Tensor>);

  void addInit(const TensorId &,
               const ONNX_NAMESPACE::TensorProto *,
               TensorType,
               const DebugInfo &di);

  Graph &graph;

  Aliases aliases;
};

} // namespace popart

#endif
