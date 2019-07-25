#define BOOST_TEST_MODULE LoggingTest

#include <boost/test/unit_test.hpp>
#include <popart/logging.hpp>

BOOST_AUTO_TEST_CASE(LoggingTest) { popart::logging::ir::debug("hello"); }
