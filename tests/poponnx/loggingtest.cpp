#define BOOST_TEST_MODULE LoggingTest

#include <boost/test/unit_test.hpp>
#include <poponnx/logging.hpp>

BOOST_AUTO_TEST_CASE(LoggingTest) { poponnx::logging::ir::debug("hello"); }
