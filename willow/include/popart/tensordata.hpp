// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_TENSORDATA_HPP_
#define POPART_WILLOW_INCLUDE_POPART_TENSORDATA_HPP_

#include <cstddef>
#include <vector>
#include <popart/error.hpp>

namespace onnx {
class TensorProto;
} // namespace onnx

namespace popart {
class TensorInfo;
class Tensor;
class Session;

// A class to hold data, used
// within the popart::Tensor class.
class TensorData {
public:
  // create by copying from src to data_,
  // the size of the copy determined by TensorInfo
  //
  // NOTE: The TensorInfo passed in here may not match the TensorInfo associated
  // with the Tensor that owns this TensorData. This is because for some
  // VariableSetting values the data owned by this class contains data for
  // multiple replica groups and this is not reflected in the TensorInfo of
  // Tensor because those describe the per-replica shapes.
  TensorData(const TensorInfo &, const void *src);

  // Instantiate TensorData with a specific size.
  TensorData(const void *src, const size_t size);

  // create by copying to data_ from ONNX_NAMESPACE::TensorProto
  TensorData(const ONNX_NAMESPACE::TensorProto &);

  void *data();
  // Expose size of data because the TensorInfo object to create this object
  // may no longer exist, and we can't rely on the Tensor's TensorInfo matching.
  size_t size() const { return data_.size(); }
  const void *data() const;

  // reset the data in the TensorData by copying from src.
  // Input data must be the same size as the existing data_
  void resetData(const TensorInfo &, const void *src);

  // Reset the data in the TensorData by copying from src.
  // Input data must be the same size as the existing data_
  // This function is used when the tensor is sharded across replicas, and the
  // tensor size is that of the full un-sharded tensor, but the data is for each
  // group's replica.
  void resetDataWithReplicaGrouping(const TensorInfo &,
                                    const void *src,
                                    int numGroups);

  // Reset the data for executablex by copying from src.
  void resetDataInExecutablex(Tensor &, Session &, const void *src);

  // reset the data in the TensorData by copying from
  // ONNX_NAMESPACE::TensorProto. Input data must be the same size as the
  // existing data_
  void resetData(const ONNX_NAMESPACE::TensorProto &);

  // reset the data in the TensorData by copying from src.
  // Input data does not have to be the same size as the
  // existing data_
  void resetDataWithNonMatchingSize(const TensorInfo &info,
                                    const std::vector<char> from);

  template <typename RESULT_TYPE>
  std::vector<RESULT_TYPE> copyDataAs(int expectedResultSize) const {
    if (data_.size() != expectedResultSize * sizeof(RESULT_TYPE)) {
      throw error("Size of data does not match expected result size. Expected "
                  "data of {} bytes, but data is {} bytes in size.",
                  expectedResultSize * sizeof(RESULT_TYPE),
                  data_.size());
    }

    std::vector<RESULT_TYPE> result;
    const RESULT_TYPE *x = reinterpret_cast<const RESULT_TYPE *>(data());
    for (int i = 0; i < expectedResultSize; i++) {
      result.push_back(x[i]);
    }
    return result;
  }

  /// Get the IsSyncedWithIPU bool. See isSyncedWithIPU.
  bool getIsSyncedWithIPU() const { return isSyncedWithIPU; }

  /// Set the IsSyncedWithIPU bool. See isSyncedWithIPU.
  void setIsSyncedWithIPU(bool val) { isSyncedWithIPU = val; }

private:
  std::vector<char> data_;

  /// Is the data stored in data_ in sync with the data on the IPU? If not a
  /// call to Devicex::readWeightsToTensorData will be required before reading
  /// the tensor's data
  bool isSyncedWithIPU = true;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_TENSORDATA_HPP_
