// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <functional>

#include <../popart/shared_cpp/np_utils.hpp>
#include <bindings/float8conversion.hpp>
#include <initializer_list>
#include <pybind11/cast.h>
#include <pybind11/functional.h> // IWYU pragma: keep
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // IWYU pragma: keep
#include <poplar/Quarter.hpp>
#include <poplar/Type.hpp>
#include <poplar/TypeConversion.hpp>
#include <popart/docs/pydocs_popart_core.hpp>
#include <popart/popx/irlowering.hpp>

namespace py = pybind11;

namespace popart {
namespace _internal {
namespace ir {

namespace {

/**
 * Convert from NumPy array of a floating point type to a NumPy array containing
 * float8 values (using the dtype `np.uint8` because float8 is not supported in
 * NumPy). This function is a host-based implementation of `(destType)(src *
 * pow2(log2Scale))`, performing a multiplication by `pow2(log2Scale)` prior to
 * casting to float8. Where values exceed the numeric range of the destination
 * type you can choose to either saturate values or produce NaNs.
 *
 * This is a PopART wrapper around Poplar's `convertToDeviceType`.
 *
 * @tparam T The floating point type to convert. Must be float or double.
 * \param destType Data type to convert to. Must be DataType::FLOAT8_143
 *   or DataType::FLOAT8_152.
 * \param src The NumPy data to convert.
 * \param log2Scale The user's data is multiplied by `pow2(log2Scale)` before
 *   casting.
 * \param notANumberOnOverflow If True produce NaN when the input values exceed
 *   the numeric range of the destination type selected. If False saturate the
 *   results.
 * \returns The converted NumPy array.
 */
template <typename T>
py::array convertToFloat8AsUInt8(const DataType destType,
                                 py::array &src,
                                 int8_t log2Scale          = 0,
                                 bool notANumberOnOverflow = true) {

  poplar::QuarterMetadata metadata;

  if (destType == DataType::FLOAT8_143) {
    // In Poplar/Poplibs, cast to quarter negates `log2Scale`.
    metadata = poplar::QuarterMetadata(poplar::QuarterMetadata::Format::F143,
                                       -log2Scale);

  } else if (destType == DataType::FLOAT8_152) {
    // In Poplar/Poplibs, cast to quarter negates `log2Scale`.
    metadata = poplar::QuarterMetadata(poplar::QuarterMetadata::Format::F152,
                                       -log2Scale);
  } else {
    throw error("Unsupported data type {} for conversion to float8", destType);
  }

  src       = makeContiguous(src);
  auto vals = static_cast<T *>(src.request().ptr);

  gccs::ArrayRef<T> ins{vals, static_cast<size_t>(src.size())};
  std::vector<uint8_t> out_vec(src.size(), 0);

  poplar::convertToDeviceType(
      poplar::QUARTER, metadata, ins, out_vec.data(), notANumberOnOverflow);

  py::array outs = py::cast(out_vec);

  return outs;
}

/**
 * Convert from a NumPy array with float8 values (with dtype `np.uint8`) to to a
 * NumPy array with dtype based on host type T (float or double). This function
 * is a host-based implementation of `(destType)(src) * pow2(log2Scale)`,
 * performing a multiplication by `pow2(log2Scale)` after casting to the
 * destination type.
 *
 * This is a PopART wrapper around Poplar's `convertFromDeviceType`.
 *
 * @tparam T The floating point type to convert to. Must be float or double.
 * \param type Data type to convert from. Must be DataType::FLOAT8_143
 *   or DataType::FLOAT8_152.
 * \param src The NumPy data to convert.
 * \param log2Scale The user's data is multiplied by `pow2(log2Scale)` after
 *   casting.
 * \returns The converted NumPy array.
 */
template <typename T>
py::array convertFromFloat8AsUInt8(const DataType srcType,
                                   py::array_t<uint8_t> &src,
                                   py::dtype &pybind_tgt_dtype,
                                   int8_t log2Scale = 0) {

  poplar::QuarterMetadata metadata;

  if (srcType == DataType::FLOAT8_143) {
    metadata = poplar::QuarterMetadata(poplar::QuarterMetadata::Format::F143,
                                       log2Scale);
  } else if (srcType == DataType::FLOAT8_152) {
    metadata = poplar::QuarterMetadata(poplar::QuarterMetadata::Format::F152,
                                       log2Scale);
  } else {
    throw error("Unsupported data type {} for conversion from float8", srcType);
  }

  src = makeContiguous(src);

  gccs::ArrayRef<uint8_t> ins{static_cast<unsigned char *>(src.request().ptr),
                              static_cast<size_t>(src.size())};

  std::vector<T> out_vec(src.size());
  gccs::ArrayRef<T> dest{out_vec};

  poplar::convertFromDeviceType(poplar::QUARTER, metadata, ins.data(), dest);

  // This does a copy of the data.
  return py::array{
      pybind_tgt_dtype, src.size(), static_cast<void *>(out_vec.data())};
}

} // namespace

void bindFloat8conversion(py::module &m) {
  {
    m.def("convertFromFloat16ToFloat8AsUInt8",
          &convertToFloat8AsUInt8<poplar::Half>);
    m.def("convertFromFloat32ToFloat8AsUInt8", &convertToFloat8AsUInt8<float>);
    m.def("convertFromFloat64ToFloat8AsUInt8", &convertToFloat8AsUInt8<double>);
    m.def("convertFromFloat8AsUInt8ToFloat16",
          &convertFromFloat8AsUInt8<poplar::Half>);
    m.def("convertFromFloat8AsUInt8ToFloat32",
          &convertFromFloat8AsUInt8<float>);
    m.def("convertFromFloat8AsUInt8ToFloat64",
          &convertFromFloat8AsUInt8<double>);
  }
}

} // namespace ir
} // namespace _internal
} // namespace popart
