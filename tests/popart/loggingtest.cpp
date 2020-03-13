// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE LoggingTest

#include <boost/test/unit_test.hpp>
#include <popart/logging.hpp>

BOOST_AUTO_TEST_CASE(LoggingTest_1) { popart::logging::ir::debug("hello"); }

BOOST_AUTO_TEST_CASE(LoggingTest_2) {
  popart::logging::ir::debug("the answer is {}", 42);
}

BOOST_AUTO_TEST_CASE(LoggingTest_3) {
  popart::logging::ir::debug("the answer is {} {}", 42, 24);
}

BOOST_AUTO_TEST_CASE(LoggingTest_4) {
  popart::logging::ir::debug("Pi is {}", 3.14);
}

BOOST_AUTO_TEST_CASE(LoggingTest_5) {
  std::string s = "over";
  popart::logging::ir::debug("game {} {}", s, "man");
}

BOOST_AUTO_TEST_CASE(LoggingTest_6) {
  auto origLevel = popart::logging::getLogLevel(popart::logging::Module::ir);
  popart::logging::setLogLevel(popart::logging::Module::ir,
                               popart::logging::Level::Trace);
  BOOST_CHECK_THROW(popart::logging::ir::debug("game {}"),
                    popart::logging::FormatError);
  popart::logging::setLogLevel(popart::logging::Module::ir, origLevel);
}

struct Foo {
  int i;
  int j;
};

std::ostream &operator<<(std::ostream &s, const Foo &f) {
  s << f.i << " " << f.j;
  return s;
}

BOOST_AUTO_TEST_CASE(LoggingTest_7) {
  Foo f;
  f.i = 200;
  f.j = 600;
  (popart::logging::ir::debug("at 12'{}'12", f));
}