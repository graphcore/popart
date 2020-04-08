// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_TENSORDATA_HPP
#define GUARD_NEURALNET_TENSORDATA_HPP

#include <functional>
#include <numeric>
#include <ostream>
#include <popart/error.hpp>
#include <popart/iarray.hpp>
#include <popart/names.hpp>
#include <popart/tensorinfo.hpp>

namespace popart {

// A class to hold data, used
// within the popart::Tensor class.
class TensorData {
public:
  // create by copying from src to data_,
  // the size of the copy determined by TensorInfo
  TensorData(const TensorInfo &, const void *src);

  // create by copying to data_ from ONNX_NAMESPACE::TensorProto
  TensorData(const ONNX_NAMESPACE::TensorProto &);

  void *data();
  const void *data() const;

  // reset the data in the TensorData by copying from src.
  // Input data must be the same size as the existing data_
  void resetData(const TensorInfo &, const void *src);

  // reset the data in the TensorData bt copying from
  // ONNX_NAMESPACE::TensorProto. Input data must be the same size as the
  // existing data_
  void resetData(const ONNX_NAMESPACE::TensorProto &);

private:
  std::vector<char> data_;
};

} // namespace popart

#endif
