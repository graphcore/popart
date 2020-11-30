// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_InputShapeInfo_HPP
#define GUARD_NEURALNET_InputShapeInfo_HPP

#include <popart/tensorinfo.hpp>

namespace popart {

// What is known about the Ir before it is run.
// This knowledge can sometimes be compiled into the Ir,
// and for certain backends is even required, for example
// the IPU requires all Stream Tensor shapes.
// In the future (TODO T5252) it will also contain indices for slicing
// tensors (I think the LSTM from pytorch might require this)
class InputShapeInfo {
public:
  InputShapeInfo() = default;

  void add(TensorId, const TensorInfo &);
  const TensorInfo &get(TensorId) const;
  bool has(TensorId) const;

  // return all unique TensorIds of tensors with any
  // information stored in this object, either TensorInfo
  // or an actual tensor
  std::vector<TensorId> getAllTensorIds() const;

  const std::map<TensorId, TensorInfo> &getInfos() const { return infos; }

private:
  std::map<TensorId, TensorInfo> infos;
  // we will also have a map of actual tensors, these
  // can be used sometimes to compile the Graph (slice
  // indices for example) (TODO T5252)
};

} // namespace popart

namespace std {
template <> struct hash<popart::InputShapeInfo> {
  std::size_t operator()(const popart::InputShapeInfo &info) const;
};
} // namespace std

namespace popart {
inline std::size_t hash_value(const InputShapeInfo &info) {
  return std::hash<InputShapeInfo>()(info);
}
} // namespace popart

#endif
