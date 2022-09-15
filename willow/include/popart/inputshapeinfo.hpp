// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_INPUTSHAPEINFO_HPP_
#define POPART_WILLOW_INCLUDE_POPART_INPUTSHAPEINFO_HPP_

#include <cstddef>
#include <functional>
#include <map>
#include <vector>
#include <popart/tensorinfo.hpp>

#include "popart/names.hpp"

namespace popart {

/**
 * Class that contains what is known about the input tensors (as TensorInfo
 * objects) in the IR prior to compilation.
 *
 * This knowledge can sometimes be compiled into the IR, and for certain
 * backends is even required, for example the IPU requires all Stream Tensor
 * shapes.
 */
class InputShapeInfo {
public:
  /**
   * Default constructor for the InputShapeInfo class.
   */
  InputShapeInfo() = default;

  /**
   * Add the identifier and TensorInfo object for a tensor to the
   * InputShapeInfo object.
   *
   * \param TensorId The identifier of the tensor for which information is being
   *     added.
   * \param TensorInfo The tensor information to be added.
   */
  void add(TensorId, const TensorInfo &);

  /**
   * Get the information of a tensor.
   *
   * \param TensorId The identifier of the tensor for which to get the tensor
   *     information.
   */
  const TensorInfo &get(TensorId) const;

  /**
   * Check if the InputShapeInfo object contains information for a tensor.
   *
   * \param TensorId The identifier of the tensor to check.
   *
   * \returns If `true`, the InputShapeInfo object contains information for the
   *     tensor. If `false`, the InputShapeInfo object does not contain
   *     information for the tensor.
   */
  bool has(TensorId) const;

  /**
   * Get all unique tensor identifiers of tensors in the InputShapeInfo object.
   *
   * \returns Vector of tensor identifiers.
   */
  std::vector<TensorId> getAllTensorIds() const;

  /**
   * Get all information contained the InputShapeInfo object.
   *
   * \returns Map of tensor identifiers and the corresponding tensor
   * information.
   */
  const std::map<TensorId, TensorInfo> &getInfos() const { return infos; }

private:
  std::map<TensorId, TensorInfo> infos;
  // we will also have a map of actual tensors, these
  // can be used sometimes to compile the Graph (slice
  // indices for example) (TODO T5284)
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

#endif // POPART_WILLOW_INCLUDE_POPART_INPUTSHAPEINFO_HPP_
