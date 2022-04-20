// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE StepIOTest

#include <algorithm>
#include <boost/test/unit_test.hpp>
#include <cstdint>
#include <initializer_list>
#include <list>
#include <memory>
#include <stepiosplitter.hpp>
#include <string>
#include <tuple>
#include <vector>

#include "popart/datatype.hpp"
#include "popart/istepio.hpp"
#include "popart/logging.hpp"
#include "popart/names.hpp"
#include "popart/tensorinfo.hpp"
#include "popart/voiddata.hpp"

using namespace popart;

namespace popart {
namespace popx {
class Executablex;
}
} // namespace popart

namespace {

// Test class that can be plugged in as a 'fake' upstream class into
// StepIOSplitter and test the splitter's interaction with it. Note
// that this could be better implemented with a mocking framework
// but I'm avoiding the cost of adding a dependency for now.
class TestStepIO : public IStepIO {
public:
  // Structs for storing params of function CallHistory.
  struct InCallParams {
    TensorId id;
    int64_t numElements;
    bool prefetch;
  };
  using InCallParamsVec = std::vector<InCallParams>;
  struct InCompleteCallParams {
    TensorId id;
    int64_t numElements;
  };
  using InCompleteCallParamsVec = std::vector<InCompleteCallParams>;
  struct OutCallParams {
    TensorId id;
    int64_t numElements;
  };
  using OutCallParamsVec = std::vector<OutCallParams>;
  struct OutCompleteCallParams {
    TensorId id;
  };
  using OutCompleteCallParamsVec = std::vector<OutCompleteCallParams>;

  TestStepIO()
      : inCallHistory{}, inCompleteCallHistory{}, outCallHistory{},
        outCompleteCallHistory{}, inReturnValues{}, outReturnValues{} {}

  virtual ~TestStepIO() = default;

  virtual ConstVoidData in(TensorId id, int64_t numElements, bool prefetch) {
    // Remember call (so we can use it in tests).
    inCallHistory.push_back(InCallParams{id, numElements, prefetch});
    BOOST_REQUIRE(!inReturnValues.empty());
    auto res = inReturnValues.front();
    logging::debug("upstream -   in({},{},{}) (return:{})",
                   id,
                   numElements,
                   prefetch,
                   res.data);
    inReturnValues.pop_front();
    return res;
  }

  virtual void inComplete(TensorId id, int64_t numElements) {
    // Remember call (so we can use it in tests).
    inCompleteCallHistory.push_back(InCompleteCallParams{id, numElements});
    logging::debug("upstream -   inComplete({},{})", id, numElements);
  }

  virtual MutableVoidData out(TensorId id, int64_t numElements) {
    // Remember call (so we can use it in tests).
    outCallHistory.push_back(OutCallParams{id, numElements});
    BOOST_REQUIRE(!outReturnValues.empty());
    auto res = outReturnValues.front();
    logging::debug(
        "upstream -   out({},{}) (return:{})", id, numElements, res.data);
    outReturnValues.pop_front();
    return res;
  }

  virtual void outComplete(TensorId id) {
    outCompleteCallHistory.push_back(OutCompleteCallParams{id});
    logging::debug("upstream -   outComplete({})", id);
  }

  virtual void assertNumElements(const popx::Executablex &) const {
    // pass
  }

  void clearHistory() {
    inCallHistory.clear();
    inCompleteCallHistory.clear();
    outCallHistory.clear();
    outCompleteCallHistory.clear();
  }

  // Parameters of CallHistory to functions.
  std::vector<InCallParams> inCallHistory;
  std::vector<InCompleteCallParams> inCompleteCallHistory;
  std::vector<OutCallParams> outCallHistory;
  std::vector<OutCompleteCallParams> outCompleteCallHistory;

  // Values to return.
  std::list<ConstVoidData> inReturnValues;
  std::list<MutableVoidData> outReturnValues;
};

enum class TestDirection {
  In  = 0,
  Out = 1,
};

std::string testDirection2Str(TestDirection &dir) {
  if (dir == TestDirection::In) {
    return "In";
  } else {
    return "Out";
  }
}

TensorId tensorId{"testTensor1"};
TensorInfo tensorInfo{DataType::FLOAT, Shape{1}};
int64_t tensorNelms{tensorInfo.nelms()};

auto setupSplitter = [](int repl, int bsp, int accum, int runs = 1) {
  const unsigned numData = repl * bsp * accum * runs;

  auto splitter = std::make_shared<StepIOSplitter>(
      repl,
      [&](const TensorId &id) { return bsp * accum; },
      [&](const TensorId &id) { return bsp * accum; });
  auto upstream    = std::make_shared<TestStepIO>();
  auto downstreams = std::vector<IStepIO *>();
  auto inbuf       = std::make_shared<std::vector<unsigned>>(numData, 0u);
  auto outbuf      = std::make_shared<std::vector<unsigned>>(numData, 0u);

  // Link upstream IStepIO to splitter.
  splitter->setUpstreamIo(upstream.get());

  // Set upstream return values (for testing).
  for (int i = 0; i < numData; ++i) {
    upstream->inReturnValues.push_back(
        ConstVoidData{reinterpret_cast<void *>(&(*inbuf)[i]), tensorInfo});
    logging::debug(
        "buffer in[{}]  - {}", i, reinterpret_cast<void *>(&(*inbuf)[i]));
  }
  for (int i = 0; i < numData; ++i) {
    upstream->outReturnValues.push_back(
        MutableVoidData{reinterpret_cast<void *>(&(*outbuf)[i]), tensorInfo});
    logging::debug(
        "buffer out[{}] - {}", i, reinterpret_cast<void *>(&(*outbuf)[i]));
  }

  // Get downstream IStepIO adapters from splitter.
  for (unsigned r = 0; r < repl; ++r) {
    downstreams.push_back(
        splitter->getDownstreamStepIO(tensorId, tensorInfo, r));
  }

  return std::make_tuple(splitter, upstream, downstreams, inbuf, outbuf);
};

// Create function that returns buffer addresses for a given offset.
auto getAddr = [](auto &setup, TestDirection &dir) {
  auto addr = [&](int i) {
    if (dir == TestDirection::In) {
      auto inbuf = std::get<3>(setup);
      // logging::debug("getAddr (inbuf[{}] == {})", i, reinterpret_cast<void
      // *>(&(*inbuf)[i]));
      return reinterpret_cast<void *>(&(*inbuf)[i]);
    } else {
      auto outbuf = std::get<4>(setup);
      // logging::debug("getAddr (outbuf[{}] == {})", i, reinterpret_cast<void
      // *>(&(*outbuf)[i]));
      return reinterpret_cast<void *>(&(*outbuf)[i]);
    }
  };
  return addr;
};

// Create function that calls 'in' on downstream IStepIO and checks
// the returned buffer matches the parameter.
auto getCallDownstream = [](auto &downstreams, TestDirection &dir) {
  auto callDownstream = [&](int repl, bool prefetch, void *expectedAddr) {
    if (dir == TestDirection::In) {
      logging::debug("downstream - in({},{},{}):{} (exp:{})",
                     tensorId,
                     tensorNelms,
                     prefetch,
                     repl,
                     expectedAddr);
      auto result = downstreams[repl]->in(tensorId, tensorNelms, prefetch);
      BOOST_CHECK_EQUAL(result.data, expectedAddr);
    } else {
      logging::debug("downstream - out({},{}):{} (exp:{})",
                     tensorId,
                     tensorNelms,
                     repl,
                     expectedAddr);
      auto result = downstreams[repl]->out(tensorId, tensorNelms);
      BOOST_CHECK_EQUAL(result.data, expectedAddr);
    }
  };
  return callDownstream;
};

// Create function that calls 'inComplete' on downstream IStepIO.
auto getCallDownstreamComplete = [](auto &downstreams, TestDirection &dir) {
  auto callDownstreamComplete = [&](int repl) {
    if (dir == TestDirection::In) {
      logging::debug(
          "downstream - inComplete({},{}):{}", tensorId, tensorNelms, repl);
      downstreams[repl]->inComplete(tensorId, tensorNelms);
    } else {
      logging::debug("downstream - outComplete({}):{}", tensorId, repl);
      downstreams[repl]->outComplete(tensorId);
    }
  };
  return callDownstreamComplete;
};

// Create function to check and clear upstream call history.
auto getCheckUpstreamCallHistory = [](auto &upstream, TestDirection &dir) {
  auto checkUpstreamCallHistory = [&](const int numBufRequests,
                                      const int numCompletes) {
    if (dir == TestDirection::In) {
      BOOST_CHECK_EQUAL(upstream->inCallHistory.size(), numBufRequests);
      BOOST_REQUIRE(upstream->inCallHistory.size() >= numBufRequests);
      for (int i = 0; i < numBufRequests; ++i) {
        BOOST_CHECK_EQUAL(upstream->inCallHistory[i].id, tensorId);
        BOOST_CHECK_EQUAL(upstream->inCallHistory[i].numElements, tensorNelms);
        BOOST_CHECK_EQUAL(upstream->inCallHistory[i].prefetch,
                          i == numBufRequests - 1);
      }
      BOOST_CHECK_EQUAL(upstream->inCompleteCallHistory.size(), numCompletes);
      BOOST_REQUIRE(upstream->inCompleteCallHistory.size() >= numCompletes);
      for (int i = 0; i < numCompletes; ++i) {
        BOOST_CHECK_EQUAL(upstream->inCompleteCallHistory[i].id, tensorId);
        BOOST_CHECK_EQUAL(upstream->inCompleteCallHistory[i].numElements,
                          tensorNelms);
      }
      BOOST_CHECK_EQUAL(upstream->outCallHistory.size(), 0);
      BOOST_CHECK_EQUAL(upstream->outCompleteCallHistory.size(), 0);
    } else {
      BOOST_CHECK_EQUAL(upstream->inCallHistory.size(), 0);
      BOOST_CHECK_EQUAL(upstream->inCompleteCallHistory.size(), 0);
      BOOST_CHECK_EQUAL(upstream->outCallHistory.size(), numBufRequests);
      BOOST_REQUIRE(upstream->outCallHistory.size() >= numBufRequests);
      for (int i = 0; i < numBufRequests; ++i) {
        BOOST_CHECK_EQUAL(upstream->outCallHistory[i].id, tensorId);
        BOOST_CHECK_EQUAL(upstream->outCallHistory[i].numElements, tensorNelms);
      }
      BOOST_CHECK_EQUAL(upstream->outCompleteCallHistory.size(), numCompletes);
      BOOST_REQUIRE(upstream->outCompleteCallHistory.size() >= numCompletes);
      for (int i = 0; i < numCompletes; ++i) {
        BOOST_CHECK_EQUAL(upstream->outCompleteCallHistory[i].id, tensorId);
      }
    }
    upstream->clearHistory();
  };
  return checkUpstreamCallHistory;
};
} // namespace

BOOST_AUTO_TEST_CASE(StepIOSplitter_prefetch_bufferDepth1_scenario1) {

  // Test the case where poplar calls in/inComplete on each replica in order,
  // e.g.:
  //
  //    POPLAR                        UPSTREAM
  //    * in (repl=0)                 in
  //    * inComplete (repl=0)         inComplete
  //    * in (repl=1)                 in
  //    * inComplete (repl=1)         inComplete
  //    * ...
  //    * in (repl=repl-1)            in
  //    * inComplete (repl=repl-1)    inComplete
  //    * etc.
  //
  auto test = [=](int repl, int bsp, int accum, TestDirection dir) {
    logging::debug("StepIOSplitter_prefetch_bufferDepth1_scenario1 - repl={}, "
                   "bsp={}, accum={}, dir={}",
                   repl,
                   bsp,
                   accum,
                   testDirection2Str(dir));

    auto setup = setupSplitter(repl, bsp, accum);

    auto splitter    = std::get<0>(setup);
    auto upstream    = std::get<1>(setup);
    auto downstreams = std::get<2>(setup);

    auto addr                     = getAddr(setup, dir);
    auto callDownstream           = getCallDownstream(downstreams, dir);
    auto callDownstreamComplete   = getCallDownstreamComplete(downstreams, dir);
    auto checkUpstreamCallHistory = getCheckUpstreamCallHistory(upstream, dir);

    int addrOffset = 0;

    for (int b = 0; b < bsp; ++b) {
      for (int a = 0; a < accum; ++a) {
        for (int r = 0; r < repl; ++r) {
          callDownstream(r, true, addr(addrOffset++));
          checkUpstreamCallHistory(/* num in/out calls= */ 1,
                                   /* num inComplete/outComplete calls= */ 0);
          callDownstreamComplete(r);
          checkUpstreamCallHistory(/* num in/out calls= */ 0,
                                   /* num inComplete/outComplete calls= */ 1);
        }
      }
    }

    // Check that once we run out of data we return nullptr and nothing gets
    // called upstream. Output can't overrun.
    if (dir == TestDirection::In) {
      for (int r = 0; r < repl; ++r) {
        callDownstream(r, true, nullptr);
        checkUpstreamCallHistory(/* num in/out calls= */ 0,
                                 /* num inComplete/outComplete calls= */ 0);
      }
    }
  };

  for (auto dir : {TestDirection::In, TestDirection::Out}) {
    test(/*repl= */ 1, /* bsp= */ 15, /*accum= */ 2, dir);
    test(/*repl= */ 2, /* bsp= */ 3, /*accum= */ 2, dir);
    test(/*repl= */ 4, /* bsp= */ 3, /*accum= */ 16, dir);
  }
}

BOOST_AUTO_TEST_CASE(StepIOSplitter_prefetch_bufferDepth1_scenario2) {

  // Test the case where poplar calls in on each replica in order and then calls
  // inComplete on each replica in order, e.g.:
  //
  //    POPLAR                        UPSTREAM
  //    * in (repl=0)                 in
  //    * in (repl=1)                 in
  //    * ...
  //    * in (repl=repl-1)            in
  //    * inComplete (repl=0)         inComplete
  //    * inComplete (repl=1)         inComplete
  //    * ...
  //    * inComplete (repl=repl-1)    inComplete
  //    * etc.
  //
  auto test = [=](int repl, int bsp, int accum, TestDirection dir) {
    logging::debug("StepIOSplitter_prefetch_bufferDepth1_scenario2 - repl={}, "
                   "bsp={}, accum={}, dir={}",
                   repl,
                   bsp,
                   accum,
                   testDirection2Str(dir));

    auto setup = setupSplitter(repl, bsp, accum);

    auto splitter    = std::get<0>(setup);
    auto upstream    = std::get<1>(setup);
    auto downstreams = std::get<2>(setup);

    auto addr                     = getAddr(setup, dir);
    auto callDownstream           = getCallDownstream(downstreams, dir);
    auto callDownstreamComplete   = getCallDownstreamComplete(downstreams, dir);
    auto checkUpstreamCallHistory = getCheckUpstreamCallHistory(upstream, dir);

    int addrOffset = 0;

    for (int b = 0; b < bsp; ++b) {
      for (int a = 0; a < accum; ++a) {
        for (int r = 0; r < repl; ++r) {
          callDownstream(r, true, addr(addrOffset++));
          checkUpstreamCallHistory(/* num in/out calls= */ 1,
                                   /* num inComplete/outComplete calls= */ 0);
        }
        for (int r = 0; r < repl; ++r) {
          callDownstreamComplete(r);
          checkUpstreamCallHistory(/* num in/out calls= */ 0,
                                   /* num inComplete/outComplete calls= */ 1);
        }
      }
    }

    // Check that once we run out of data we return nullptr and nothing gets
    // called upstream. Output can't overrun.
    if (dir == TestDirection::In) {
      for (int r = 0; r < repl; ++r) {
        callDownstream(r, true, nullptr);
        checkUpstreamCallHistory(/* num in/out calls= */ 0,
                                 /* num inComplete/outComplete calls= */ 0);
      }
    }
  };

  for (auto dir : {TestDirection::In, TestDirection::Out}) {
    test(/*repl= */ 1, /* bsp= */ 15, /*accum= */ 2, dir);
    test(/*repl= */ 2, /* bsp= */ 3, /*accum= */ 2, dir);
    test(/*repl= */ 4, /* bsp= */ 3, /*accum= */ 16, dir);
  }
}

BOOST_AUTO_TEST_CASE(StepIOSplitter_prefetch_bufferDepth1_scenario3) {

  // Test the case where poplar calls replicas out of order:
  //
  //    POPLAR                        UPSTREAM
  //    * in (repl=2)                 in, in, in
  //    * in (repl=1)                 -
  //    * inComplete(repl=2)          -
  //    * in (repl=0)                 -
  //    * inComplete(repl=0)          inComplete
  //    * in (repl=3)                 in
  //    * inComplete(repl=1)          inComplete, inComplete
  //    * inComplete(repl=3)          inComplete
  //    * repeated.
  //
  auto test = [=](int bsp, int accum, TestDirection dir) {
    const int repl = 4;

    logging::debug("StepIOSplitter_prefetch_bufferDepth1_scenario3 - repl={}, "
                   "bsp={}, accum={}, dir={}",
                   repl,
                   bsp,
                   accum,
                   testDirection2Str(dir));

    auto setup = setupSplitter(repl, bsp, accum);

    auto splitter    = std::get<0>(setup);
    auto upstream    = std::get<1>(setup);
    auto downstreams = std::get<2>(setup);

    auto addr                     = getAddr(setup, dir);
    auto callDownstream           = getCallDownstream(downstreams, dir);
    auto callDownstreamComplete   = getCallDownstreamComplete(downstreams, dir);
    auto checkUpstreamCallHistory = getCheckUpstreamCallHistory(upstream, dir);

    int addrOffset = 0;

    for (int b = 0; b < bsp; ++b) {
      for (int a = 0; a < accum; ++a) {

        callDownstream(2, true, addr(addrOffset + 2));
        checkUpstreamCallHistory(/* num in/out calls= */ 3,
                                 /* num inComplete/outComplete calls= */ 0);
        callDownstream(1, true, addr(addrOffset + 1));
        checkUpstreamCallHistory(/* num in/out calls= */ 0,
                                 /* num inComplete/outComplete calls= */ 0);
        callDownstreamComplete(2);
        checkUpstreamCallHistory(/* num in/out calls= */ 0,
                                 /* num inComplete/outComplete calls= */ 0);
        callDownstream(0, true, addr(addrOffset + 0));
        checkUpstreamCallHistory(/* num in/out calls= */ 0,
                                 /* num inComplete/outComplete calls= */ 0);
        callDownstreamComplete(0);
        checkUpstreamCallHistory(/* num in/out calls= */ 0,
                                 /* num inComplete/outComplete calls= */ 1);
        callDownstream(3, true, addr(addrOffset + 3));
        checkUpstreamCallHistory(/* num in/out calls= */ 1,
                                 /* num inComplete/outComplete calls= */ 0);
        callDownstreamComplete(1);
        checkUpstreamCallHistory(/* num in/out calls= */ 0,
                                 /* num inComplete/outComplete calls= */ 2);
        callDownstreamComplete(3);
        checkUpstreamCallHistory(/* num in/out calls= */ 0,
                                 /* num inComplete/outComplete calls= */ 1);

        addrOffset += repl;
      }
    }

    // Check that once we run out of data we return nullptr and nothing gets
    // called upstream. Output can't overrun.
    if (dir == TestDirection::In) {
      for (int r = 0; r < repl; ++r) {
        callDownstream(r, true, nullptr);
        checkUpstreamCallHistory(/* num in/out calls= */ 0,
                                 /* num inComplete/outComplete calls= */ 0);
      }
    }
  };

  for (auto dir : {TestDirection::In, TestDirection::Out}) {
    test(/* bsp= */ 15, /*accum= */ 2, dir);
  }
}

BOOST_AUTO_TEST_CASE(StepIOSplitter_prefetch_bufferDepth_1_2_3_scenario1) {

  // Test the case where poplar calls in/inComplete on each replica in order,
  // e.g. for bufferingDepth=3:
  //
  //    POPLAR                        UPSTREAM
  //    * in (repl=0)                 in
  //    * ...
  //    * in (repl=repl-1)            in
  //    * in (repl=0)                 in
  //    * ...
  //    * in (repl=repl-1)            in
  //    * in (repl=0)                 in
  //    * ...
  //    * in (repl=repl-1)            in
  //    * inComplete (repl=1)         inComplete
  //    * ...
  //    * inComplete (repl=repl-1)    inComplete
  //    * in (repl=0)                 in
  //    * ...
  //    * in (repl=repl-1)            in
  //    * inComplete (repl=1)         inComplete
  //    * ...
  //    * inComplete (repl=repl-1)    inComplete
  //    * in (repl=0)                 in
  //    * ...
  //    * in (repl=repl-1)            in
  //    * etc.
  //
  auto test = [=](int repl,
                  int bsp,
                  int accum,
                  int bufferingDepth,
                  TestDirection dir) {
    logging::debug("StepIOSplitter_prefetch_bufferDepth_1_2_3_scenario1 - "
                   "repl={}, bsp={}, accum={}, bufDepth={}, dir={}",
                   repl,
                   bsp,
                   accum,
                   bufferingDepth,
                   testDirection2Str(dir));

    auto setup = setupSplitter(repl, bsp, accum);

    auto splitter    = std::get<0>(setup);
    auto upstream    = std::get<1>(setup);
    auto downstreams = std::get<2>(setup);

    auto addr                     = getAddr(setup, dir);
    auto callDownstream           = getCallDownstream(downstreams, dir);
    auto callDownstreamComplete   = getCallDownstreamComplete(downstreams, dir);
    auto checkUpstreamCallHistory = getCheckUpstreamCallHistory(upstream, dir);

    int addrOffset = 0;
    int round      = 0;

    for (int b = 0; b < bsp; ++b) {
      for (int a = 0; a < accum; ++a) {
        for (int r = 0; r < repl; ++r) {
          callDownstream(r, true, addr(addrOffset++));
          checkUpstreamCallHistory(/* num in/out calls= */ 1,
                                   /* num inComplete/outComplete calls= */ 0);

          if (round >= (bufferingDepth - 1)) {
            // Don't run for the first (bufferingDepth - 1) rounds.
            callDownstreamComplete(r);
            checkUpstreamCallHistory(/* num in/out calls= */ 0,
                                     /* num inComplete/outComplete calls= */ 1);
          }
        }
        round++;
      }
    }

    // Complete the outstanding buffers.
    for (int b = 0; b < bufferingDepth - 1; ++b) {
      for (int r = 0; r < repl; ++r) {
        callDownstreamComplete(r);
        checkUpstreamCallHistory(/* num in/out calls= */ 0,
                                 /* num inComplete/outComplete calls= */ 1);
      }
    }

    // Check that once we run out of data we return nullptr and nothing gets
    // called upstream. Output can't overrun.
    if (dir == TestDirection::In) {
      for (int r = 0; r < repl; ++r) {
        callDownstream(r, true, nullptr);
        checkUpstreamCallHistory(/* num in/out calls= */ 0,
                                 /* num inComplete/outComplete calls= */ 0);
      }
    }
  };

  for (auto dir : {TestDirection::In, TestDirection::Out}) {
    for (auto bd : {1, 2, 3}) {
      test(/*repl= */ 1,
           /* bsp= */ 7,
           /*accum= */ 2,
           /*bufferingDepth=*/bd,
           dir);
      test(/*repl= */ 2,
           /* bsp= */ 3,
           /*accum= */ 1,
           /*bufferingDepth=*/bd,
           dir);
      test(/*repl= */ 4,
           /* bsp= */ 3,
           /*accum= */ 2,
           /*bufferingDepth=*/bd,
           dir);
    }
  }
}

BOOST_AUTO_TEST_CASE(StepIOSplitter_reset) {

  // Test it's possible to re-user the splitter following a reset.
  auto test = [=](int repl, int bsp, int accum, int runs, TestDirection dir) {
    logging::debug("StepIOSplitter_reset - repl={}, "
                   "bsp={}, accum={}, runs={}, dir={}",
                   repl,
                   bsp,
                   accum,
                   runs,
                   testDirection2Str(dir));

    auto setup = setupSplitter(repl, bsp, accum, runs);

    auto splitter    = std::get<0>(setup);
    auto upstream    = std::get<1>(setup);
    auto downstreams = std::get<2>(setup);

    auto addr                     = getAddr(setup, dir);
    auto callDownstream           = getCallDownstream(downstreams, dir);
    auto callDownstreamComplete   = getCallDownstreamComplete(downstreams, dir);
    auto checkUpstreamCallHistory = getCheckUpstreamCallHistory(upstream, dir);

    int addrOffset = 0;

    for (int k = 0; k < runs; k++) {
      for (int b = 0; b < bsp; ++b) {
        for (int a = 0; a < accum; ++a) {
          for (int r = 0; r < repl; ++r) {
            callDownstream(r, true, addr(addrOffset++));
            checkUpstreamCallHistory(/* num in/out calls= */ 1,
                                     /* num inComplete/outComplete calls= */ 0);
            callDownstreamComplete(r);
            checkUpstreamCallHistory(/* num in/out calls= */ 0,
                                     /* num inComplete/outComplete calls= */ 1);
          }
        }
      }

      // Check that once we run out of data we return nullptr and nothing gets
      // called upstream. Output can't overrun.
      if (dir == TestDirection::In) {
        for (int r = 0; r < repl; ++r) {
          callDownstream(r, true, nullptr);
          checkUpstreamCallHistory(/* num in/out calls= */ 0,
                                   /* num inComplete/outComplete calls= */ 0);
        }
      }

      splitter->reset();
    }
  };

  for (auto dir : {TestDirection::In, TestDirection::Out}) {
    test(/*repl= */ 4, /* bsp= */ 3, /*accum= */ 16, /*runs=*/2, dir);
  }
}
