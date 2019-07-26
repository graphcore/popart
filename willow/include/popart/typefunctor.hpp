#ifndef GUARD_TYPEFUNCTOR_HPP
#define GUARD_TYPEFUNCTOR_HPP

#include <popart/error.hpp>
#include <popart/half.hpp>
#include <popart/names.hpp>
#include <popart/tensorinfo.hpp>

namespace popart {
namespace typefunctor {

template <typename Functor, typename ReturnType, typename... Args>
ReturnType get(DataType dtype, Args &&... args) {
  switch (dtype) {
  case DataType::DOUBLE:
    return Functor().template operator()<double>(std::forward<Args>(args)...);
  case DataType::FLOAT:
    return Functor().template operator()<float>(std::forward<Args>(args)...);
  case DataType::INT64:
    return Functor().template operator()<int64_t>(std::forward<Args>(args)...);
  case DataType::INT32:
    return Functor().template operator()<int32_t>(std::forward<Args>(args)...);
  case DataType::INT16:
    return Functor().template operator()<int16_t>(std::forward<Args>(args)...);
  case DataType::INT8:
    return Functor().template operator()<int8_t>(std::forward<Args>(args)...);
  case DataType::UINT64:
    return Functor().template operator()<uint64_t>(std::forward<Args>(args)...);
  case DataType::UINT32:
    return Functor().template operator()<uint32_t>(std::forward<Args>(args)...);
  case DataType::UINT16:
    return Functor().template operator()<uint16_t>(std::forward<Args>(args)...);
  case DataType::UINT8:
    return Functor().template operator()<uint8_t>(std::forward<Args>(args)...);
  case DataType::FLOAT16:
    return Functor().template operator()<float16_t>(
        std::forward<Args>(args)...);
  case DataType::BOOL:
  case DataType::BFLOAT16:
  case DataType::COMPLEX64:
  case DataType::COMPLEX128:
  case DataType::STRING:
  case DataType::UNDEFINED:
  default:
    throw error("functor {} does not support DataType::{}",
                typeid(Functor).name(),
                getDataTypeInfoMap().at(dtype).name());
  }
}

class Int64FromVoid {
public:
  template <typename T> int64_t operator()(void *data) {
    // no good test we can do at this point that the cast
    // from void * to T * is valid, such tests must be done before
    // the call to this function using other data/clues
    return static_cast<int64_t>(reinterpret_cast<T *>(data)[0]);
  }
};

template <> int64_t Int64FromVoid::operator()<popart::Half>(void *);

} // namespace typefunctor
} // namespace popart

#endif
