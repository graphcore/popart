// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_TENSORDATA_HPP_
#define POPART_WILLOW_INCLUDE_POPART_TENSORDATA_HPP_

#include <cstddef>
#include <memory>
#include <vector>
#include <popart/error.hpp>

namespace onnx {
class TensorProto;
} // namespace onnx

namespace popart {
class TensorInfo;

// A class to hold data, used
// within the popart::Tensor class.
class TensorData {
private:
  class IData;
  class OwningData;
  class NonOwningData;

  TensorData(std::shared_ptr<IData>);

public:
  /**
   * \brief Factory to create a TensorData from a copy of the provided buffer.
   *
   * \param src Pointer to buffer to copy.
   * \param size Size in bytes of \p src buffer.
   * \return TensorData object that owns a copy of the buffer you provided.
   */
  static TensorData fromCopyOf(const void *src, std::size_t size);

  /**
   * \brief Factory to create a TensorData that contains a non-owning pointer to
   * the buffer you provided.
   *
   * If you create a TensorData this way, it does not copy your buffer and does
   * not own the buffer, therefore it is your responsibility to keep the buffer
   * alive and not clobber it.
   *
   * \param src Pointer to buffer that will be aliased.
   * \param size Size in bytes of \p src buffer.
   * \return TensorData object that aliases and does not own the buffer you
   * passed.
   */
  static TensorData fromViewOf(void *src, std::size_t size);

  /**
   * \brief Factory to create a TensorData by emplacement of a
   * `std::vector<char> &&`.
   *
   * This results in a TensorData that has a data buffer it owns. The \p data
   * you passed in will be moved-from.
   *
   * \param data STL container of your data buffer that will be moved-from.
   * \return TensorData TensorData object that owns the data you provided.
   */
  static TensorData fromEmplaceOf(std::vector<char> &&data);

  // Because we have a shared_ptr<IData> member, but IData is forward-declared,
  // we need to forward-declare the destructor too (and define it in the cpp
  // where ~IData is actually defined).
  ~TensorData();
  // Because we have a user-defined destructor, we must manually declare all
  // copy/move functions.
  TensorData(const TensorData &other);
  TensorData(TensorData &&other) noexcept;
  TensorData &operator=(const TensorData &other);
  TensorData &operator=(TensorData &&other) noexcept;

  void *data();
  // Expose size of data because the TensorInfo object to create this object
  // may no longer exist, and we can't rely on the Tensor's TensorInfo matching.
  std::size_t size() const;
  const void *data() const;

  // reset the data in the TensorData by copying from src.
  // Input data must be the same size as the existing data_
  //
  // Developer note: You cannot implement a resetDataFromEmplaceOf as the
  // the pointer returned by data() CANNOT be changed, because the Poplar
  // streams are fixed to operate on that pointer.
  void resetData(const TensorInfo &, const void *src);

  // Reset the data in the TensorData by copying from src.
  // Input data must be the same size as the existing data_
  // This function is used when the tensor is sharded across replicas, and the
  // tensor size is that of the full un-sharded tensor, but the data is for each
  // group's replica.
  void resetDataWithReplicaGrouping(const TensorInfo &,
                                    const void *src,
                                    int numGroups);

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
    if (size() != expectedResultSize * sizeof(RESULT_TYPE)) {
      throw error("Size of data does not match expected result size. Expected "
                  "data of {} bytes, but data is {} bytes in size.",
                  expectedResultSize * sizeof(RESULT_TYPE),
                  size());
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
  // shared_ptr so TensorData is copyable.
  std::shared_ptr<IData> data_;

  /// Is the data stored in data_ in sync with the data on the IPU? If not a
  /// call to Devicex::readWeightsToTensorData will be required before reading
  /// the tensor's data
  bool isSyncedWithIPU = true;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_TENSORDATA_HPP_
