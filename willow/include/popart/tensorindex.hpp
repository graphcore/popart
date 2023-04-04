// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_TENSORINDEX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_TENSORINDEX_HPP_

#include <iosfwd>
#include <map>
#include <string>
#include <vector>

#include "popart/names.hpp"
#include "popart/pointercomparators.hpp"
#include "popart/tensordebuginfo.hpp"

namespace popart {

class Tensor;
class TensorInfo;

// Inputs and outputs to Ops will use this class.
// inputs (outputs) enter (leave) at certain indices
// of an Op. There is 1 tensor per index,
// but 1+ index per tensor.
class TensorIndexMap {
public:
  TensorIndexMap() = default;
  ~TensorIndexMap();
  void insert(int, Tensor *);
  // the Tensor at index changes. Note that there
  // must already be a Tensor at the index (otherwise insert should be used)
  void reset(int, Tensor *);
  // Remove the Tensor index from the tensorMap.
  // If the Tensor is not referred to by any indices, it is removed from the
  // indicesMap.
  void erase(int);
  void clear();
  bool contains(Tensor *) const;

  // get the Tensor at index
  Tensor *tensor(int);
  const Tensor *tensor(int) const;
  // The id of the Tensor at an index
  // This is just a helper function (same as tensor(int)->id)
  TensorId id(int) const;
  bool hasIndex(int) const;
  const std::vector<int> &indices(Tensor *) const;
  const std::map<Tensor *, std::vector<int>, PTensorCmp> &indicesMap() const;
  const std::map<int, Tensor *> &tensorMap() const;
  // Unique list of tensors in the TensorIndexMap
  const std::vector<Tensor *> tensors() const;
  std::map<int, TensorId> tensorIdMap() const;
  std::map<TensorId, int> idMap() const;

  // the number of indices. Exactly the number of keys of tensor_map
  int n() const;
  void append(std::stringstream &, std::string prefix, int max_id_length) const;
  // set the TensorInfo of tensor(index) if hasIndex(index) is true
  void setInfoIfIndex(const TensorInfo &, int index);
  // the returned vector has correct TensorIds at indices in
  // tensor_map and "" at unused indices inbetween
  std::vector<TensorId> getSerialised() const;
  // returns the longest TensorId of all Tensors in indices_map
  int maxIdLength() const;
  // returns the shapes of the tensors at the indices
  std::map<int, Shape> getIndexShapeMap();
  // Return the lowest index in the map
  int minIndex() const;
  // Return the highest index in the map
  int maxIndex() const;

private:
  std::map<int, Tensor *> tensor_map;
  std::map<Tensor *, std::vector<int>, PTensorCmp> indices_map;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_TENSORINDEX_HPP_
