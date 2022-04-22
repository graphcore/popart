// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#define BOOST_TEST_MODULE CastTest
#include <boost/test/unit_test.hpp>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>
#include <popart/util.hpp>

#include "popart/datatype.hpp"
#include "popart/names.hpp"

using namespace popart;

BOOST_AUTO_TEST_CASE(Cast_Overflow) {
  int64_t val = std::numeric_limits<int64_t>::max();
  try {
    auto r = cast(DataType::INT64, DataType::INT32, &val, sizeof(int64_t));
  } catch (std::runtime_error &e) {
    BOOST_CHECK_EQUAL(e.what(),
                      "[cast] Cast int64 -> int32 failed: bad numeric "
                      "conversion: positive overflow");
  }
}

BOOST_AUTO_TEST_CASE(Cast_Unsupported) {
  int64_t val = 1357;
  try {
    auto r = cast(DataType::INT64, DataType::DOUBLE, &val, sizeof(int64_t));
  } catch (std::runtime_error &e) {
    BOOST_CHECK_EQUAL(e.what(),
                      "[cast] Unsupported cast data types int64 -> float64");
  }
}

BOOST_AUTO_TEST_CASE(Cast_Valid) {
  int64_t val = 1357;
  auto r      = cast(DataType::INT64, DataType::INT32, &val, sizeof(int64_t));
  auto cval   = *static_cast<int32_t *>(static_cast<void *>(r.data()));
  BOOST_CHECK_EQUAL(val, cval);
}
