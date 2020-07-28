// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE StepIOTest

#include <boost/test/unit_test.hpp>
#include <popart/stepiosplitter.hpp>

using namespace popart;

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
  struct InCompleteCallParams {
    TensorId id;
    int64_t numElements;
  };
  struct OutCallParams {
    TensorId id;
    int64_t numElements;
  };
  struct OutCompleteCallParams {
    TensorId id;
  };

  TestStepIO()
      : inCallHistory{}, inCompleteCallHistory{}, outCallHistory{},
        outCompleteCallHistory{}, inReturnValues{}, outReturnValues{} {}

  virtual ~TestStepIO() = default;

  virtual ConstVoidData in(TensorId id, int64_t numElements, bool prefetch) {
    // Remember call (so we can use it in tests).
    inCallHistory.push_back(InCallParams{id, numElements, prefetch});

    BOOST_ASSERT(!inReturnValues.empty());
    auto res = inReturnValues.front();
    inReturnValues.pop_front();
    return res;
  }

  virtual void inComplete(TensorId id, int64_t numElements) {
    // Remember call (so we can use it in tests).
    inCompleteCallHistory.push_back(InCompleteCallParams{id, numElements});
  }

  virtual MutableVoidData out(TensorId id, int64_t numElements) {
    // Remember call (so we can use it in tests).
    outCallHistory.push_back(OutCallParams{id, numElements});

    BOOST_ASSERT(!outReturnValues.empty());
    auto res = outReturnValues.front();
    outReturnValues.pop_front();
    return res;
  }

  virtual void outComplete(TensorId id) {
    outCompleteCallHistory.push_back(OutCompleteCallParams{id});
  }

  virtual void assertNumElements(const Ir &) const {
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

void runTestSquence(std::function<const void *(unsigned)> &fetchFn,
                    std::function<void(unsigned)> &completeFn,
                    std::function<void *(size_t)> &dataItemFn) {
  auto down0_data0 = fetchFn(0);
  auto down1_data0 = fetchFn(1);
  auto down2_data0 = fetchFn(2);
  auto down3_data0 = fetchFn(3);
  completeFn(0);
  completeFn(1);
  completeFn(2);
  completeFn(3);
  auto down0_data1 = fetchFn(0);
  auto down1_data1 = fetchFn(1);
  auto down2_data1 = fetchFn(2);
  auto down3_data1 = fetchFn(3);
  completeFn(0);
  completeFn(1);
  completeFn(2);
  completeFn(3);

  // Now try a slightly more chaotic order of downstream CallHistory.
  auto down0_data2  = fetchFn(0);
  auto down1_data2a = fetchFn(1);
  auto down1_data2b = fetchFn(1); // fetch same data twice.
  completeFn(0);                  // complete before other replicas
  auto down0_data3 = fetchFn(0);  // replica 0 on second round
  auto down3_data2 = fetchFn(3);  // out of order
  auto down2_data2 = fetchFn(2);
  completeFn(1);
  completeFn(2);
  completeFn(3);
  completeFn(0);
  auto down3_data3 = fetchFn(3);
  completeFn(3);
  auto down2_data3 = fetchFn(2);
  auto down1_data3 = fetchFn(1);
  completeFn(1);
  completeFn(2);

  // Check we got presented the data in the right order.
  BOOST_CHECK(down0_data0 == dataItemFn(0));
  BOOST_CHECK(down1_data0 == dataItemFn(1));
  BOOST_CHECK(down2_data0 == dataItemFn(2));
  BOOST_CHECK(down3_data0 == dataItemFn(3));
  BOOST_CHECK(down0_data1 == dataItemFn(4));
  BOOST_CHECK(down1_data1 == dataItemFn(5));
  BOOST_CHECK(down2_data1 == dataItemFn(6));
  BOOST_CHECK(down3_data1 == dataItemFn(7));
  BOOST_CHECK(down0_data2 == dataItemFn(8));
  BOOST_CHECK(down1_data2a == dataItemFn(9)); // Got same data twice.
  BOOST_CHECK(down1_data2b == dataItemFn(9));
  BOOST_CHECK(down2_data2 == dataItemFn(10));
  BOOST_CHECK(down3_data2 == dataItemFn(11));
  BOOST_CHECK(down0_data3 == dataItemFn(12));
  BOOST_CHECK(down1_data3 == dataItemFn(13));
  BOOST_CHECK(down2_data3 == dataItemFn(14));
  BOOST_CHECK(down3_data3 == dataItemFn(15));
}

BOOST_AUTO_TEST_CASE(StepIOSplitter_Order) {

  const size_t NUM_INPUTS     = 100;
  const unsigned NUM_REPLICAS = 4;

  TensorId tensorId{"testTensor1"};
  TensorInfo tensorInfo{DataType::FLOAT, Shape{1}};
  int64_t tensorNelms{tensorInfo.nelms()};

  // Data buffer we can use for comparing memory locations.
  std::vector<unsigned> inDataBuffer{NUM_INPUTS, 0u};
  std::vector<unsigned> outDataBuffer{NUM_INPUTS, 0u};

  // Set up upstream IStepIO.
  TestStepIO upstreamIo;

  for (int i = 0; i < NUM_INPUTS; ++i) {
    ConstVoidData inData;
    inData.data = static_cast<void *>(&inDataBuffer[i]);
    inData.info = tensorInfo;
    upstreamIo.inReturnValues.push_back(inData);

    MutableVoidData outData;
    outData.data = static_cast<void *>(&outDataBuffer[i]);
    outData.info = tensorInfo;
    upstreamIo.outReturnValues.push_back(outData);
  }

  // Set up StepIOSplitter.
  StepIOSplitter splitter(NUM_REPLICAS);
  splitter.setUpstreamIo(&upstreamIo);

  // Get downstream StepIOs from splitter.
  std::vector<IStepIO *> downstreamIos;
  for (unsigned replicationIndex = 0; replicationIndex < NUM_REPLICAS;
       ++replicationIndex) {
    downstreamIos.push_back(
        splitter.getDownstreamStepIO(tensorId, tensorInfo, replicationIndex));
  }

  // Get downstream Ios again for some replicas to ensure it won't cause
  // problems.
  splitter.getDownstreamStepIO(tensorId, tensorInfo, 0);
  splitter.getDownstreamStepIO(tensorId, tensorInfo, 3);

  // Define helper functions for testing.
  std::function<const void *(unsigned)> inFetch =
      [&](unsigned replicationIndex) -> const void * {
    return downstreamIos[replicationIndex]
        ->in(tensorId, tensorNelms, false)
        .data;
  };
  std::function<void(unsigned)> inComplete =
      [&](unsigned replicationIndex) -> void {
    downstreamIos[replicationIndex]->inComplete(tensorId, tensorNelms);
  };
  std::function<void *(size_t)> inDataItem = [&](size_t index) -> void * {
    return static_cast<void *>(&inDataBuffer[index]);
  };

  // Run tests for inputs.
  runTestSquence(inFetch, inComplete, inDataItem);

  // Define helper functions for testing.
  std::function<const void *(unsigned)> outFetch =
      [&](unsigned replicationIndex) -> const void * {
    return downstreamIos[replicationIndex]->out(tensorId, tensorNelms).data;
  };
  std::function<void(unsigned)> outComplete =
      [&](unsigned replicationIndex) -> void {
    downstreamIos[replicationIndex]->outComplete(tensorId);
  };
  std::function<void *(size_t)> outDataItem = [&](size_t index) -> void * {
    return static_cast<void *>(&outDataBuffer[index]);
  };

  // Run tests for outputs.
  runTestSquence(outFetch, outComplete, outDataItem);
}

BOOST_AUTO_TEST_CASE(StepIOSplitter_PrefetchFlag) {

  //  IStepIO         StepIOSplitter           Poplar
  // (upstream)        (under test)          (downstream)
  //    |                   |                     |
  //    | [STEP 1]          |                     |
  //    |                   |                     |
  //    |                   | <--in[0](..,true)-- |
  //    | <--in(..,true)--- |                     |
  //    | ---ret:nullptr--> |                     |
  //    |                   | ---ret:nullptr----> |
  //    |                   |                     |
  //    | [STEP 2]          |                     |
  //    |                   |                     |
  //    |                   | <--in[0](..,true)-- |
  //    | <--in(..,true)--- |                     |
  //    | ---ret:&[4]-----> |                     |
  //    | <--inComplete(..) |                     |
  //    | ----------------> |                     |
  //    |                   | ---ret:&[4]-------> |
  //    |                   |                     |
  //    | [STEP 3]          |                     |
  //    |                   |                     |
  //    |                   | <--in[2](..,true)-- |
  //    | <--in(..,false)-- |                     |
  //    | ---ret:&[7]-----> |                     |
  //    | <--inComplete(..) |                     |
  //    | ----------------> |                     |
  //    | <--in(..,true)--- |                     |
  //    | ---ret:&[8]-----> |                     |
  //    | <--inComplete(..) |                     |
  //    | ----------------> |                     |
  //    |                   | ---ret:&[8]-------> |
  //    |                   |                     |
  //    | [STEP 4]          |                     |
  //    |                   |                     |
  //    |                   | <--in[1](..,true)-- |
  //    |                   | ---ret:&[7]-------> |
  //    |                   |                     |
  //    | [STEP 5]          |                     |
  //    |                   |                     |
  //    |                   | <--inComplete[1]()- |
  //    |                   | ------------------> |
  //    |                   | <--in[1](..,false)- |
  //    | <--in(..,false)-- |                     |
  //    | ---ret:&[11]----> |                     |
  //    | <--inComplete(..) |                     |
  //    | ----------------> |                     |
  //    | <--in(..,false)-- |                     |
  //    | ---ret:&[16]----> |                     |
  //    | <--inComplete(..) |                     |
  //    | ----------------> |                     |
  //    | <--in(..,false)-- |                     |
  //    | ---ret:&[17]----> |                     |
  //    | <--inComplete(..) |                     |
  //    | ----------------> |                     |
  //    |                   | ---ret:&[17]------> |
  //

  const size_t NUM_INPUTS     = 100;
  const unsigned NUM_REPLICAS = 4;

  TensorId tensorId{"testTensor1"};
  TensorInfo tensorInfo{DataType::FLOAT, Shape{1}};
  int64_t tensorNelms{tensorInfo.nelms()};

  TestStepIO upstreamIo;

  // Set up StepIOSplitter.
  StepIOSplitter splitter(NUM_REPLICAS);
  splitter.setUpstreamIo(&upstreamIo);

  // Get downstream StepIOs from splitter.
  std::vector<IStepIO *> downstreamIos;
  for (unsigned replicationIndex = 0; replicationIndex < NUM_REPLICAS;
       ++replicationIndex) {
    downstreamIos.push_back(
        splitter.getDownstreamStepIO(tensorId, tensorInfo, replicationIndex));
  }

  // Data buffer we can use for comparing memory locations.
  std::vector<unsigned> inDataBuffer{NUM_INPUTS, 0u};

  ConstVoidData result;

  // STEP 1: The steps below tests a failed prefetch on replica 0.

  // Ensure history is clear.
  upstreamIo.clearHistory();
  // Tell upstreamIo what to return when called.
  upstreamIo.inReturnValues.push_back(ConstVoidData{nullptr, tensorInfo});
  // Request prefetch for first replica.
  result = downstreamIos[0]->in(tensorId, tensorNelms, /*prefetch=*/true);
  // Check that the downstream result indicated failure (as per upstream return
  // value).
  BOOST_ASSERT(result.data == nullptr);
  // Check that the upstream IStepIO was called with prefetch=true;
  BOOST_ASSERT(upstreamIo.inCallHistory.size() == 1);
  BOOST_ASSERT(upstreamIo.inCompleteCallHistory.size() == 0);
  BOOST_ASSERT(upstreamIo.outCallHistory.size() == 0);
  BOOST_ASSERT(upstreamIo.outCompleteCallHistory.size() == 0);
  BOOST_ASSERT(upstreamIo.inCallHistory[0].id == tensorId);
  BOOST_ASSERT(upstreamIo.inCallHistory[0].numElements == tensorNelms);
  BOOST_ASSERT(upstreamIo.inCallHistory[0].prefetch == true);

  // STEP 2: The steps below tests a successful prefetch on replica 0.

  // Clear history between test steps.
  upstreamIo.clearHistory();
  // Tell upstreamIo what to return when called.
  upstreamIo.inReturnValues.push_back(
      ConstVoidData{reinterpret_cast<void *>(&inDataBuffer[4]), tensorInfo});
  // Request prefetch for first replica.
  result = downstreamIos[0]->in(tensorId, tensorNelms, /*prefetch=*/true);
  // Check that the downstream result indicated success (as per upstream return
  // value).
  BOOST_ASSERT(result.data == reinterpret_cast<void *>(&inDataBuffer[4]));
  // Check that the upstream IStepIO was called with prefetch=true and that
  // complete was called at the earliest opportunity;
  BOOST_ASSERT(upstreamIo.inCallHistory.size() == 1);
  BOOST_ASSERT(upstreamIo.inCompleteCallHistory.size() == 1);
  BOOST_ASSERT(upstreamIo.outCallHistory.size() == 0);
  BOOST_ASSERT(upstreamIo.outCompleteCallHistory.size() == 0);
  BOOST_ASSERT(upstreamIo.inCallHistory[0].id == tensorId);
  BOOST_ASSERT(upstreamIo.inCallHistory[0].numElements == tensorNelms);
  BOOST_ASSERT(upstreamIo.inCallHistory[0].prefetch == true);
  BOOST_ASSERT(upstreamIo.inCompleteCallHistory[0].id == tensorId);
  BOOST_ASSERT(upstreamIo.inCompleteCallHistory[0].numElements == tensorNelms);

  // STEP 3: Next, do a prefetch on replica 2 with prefetch. This should result
  // in a non-prefetch call for replica 1 and a prefetch call for replica 2.

  // Clear history between test steps.
  upstreamIo.clearHistory();
  // Tell upstreamIo what to return when called.
  upstreamIo.inReturnValues.push_back(
      ConstVoidData{reinterpret_cast<void *>(&inDataBuffer[7]), tensorInfo});
  upstreamIo.inReturnValues.push_back(
      ConstVoidData{reinterpret_cast<void *>(&inDataBuffer[8]), tensorInfo});
  // Request prefetch for replica 2.
  result = downstreamIos[2]->in(tensorId, tensorNelms, /*prefetch=*/true);
  // Check that the downstream result indicated success (as per upstream return
  // value).
  BOOST_ASSERT(result.data == reinterpret_cast<void *>(&inDataBuffer[8]));
  // Check that the upstream IStepIO was called twice, once with prefetch=false
  // and once with prefetch=true;
  BOOST_ASSERT(upstreamIo.inCallHistory.size() == 2);
  BOOST_ASSERT(upstreamIo.inCompleteCallHistory.size() == 2);
  BOOST_ASSERT(upstreamIo.outCallHistory.size() == 0);
  BOOST_ASSERT(upstreamIo.outCompleteCallHistory.size() == 0);
  BOOST_ASSERT(upstreamIo.inCallHistory[0].id == tensorId);
  BOOST_ASSERT(upstreamIo.inCallHistory[0].numElements == tensorNelms);
  BOOST_ASSERT(upstreamIo.inCallHistory[0].prefetch == false);
  BOOST_ASSERT(upstreamIo.inCallHistory[1].id == tensorId);
  BOOST_ASSERT(upstreamIo.inCallHistory[1].numElements == tensorNelms);
  BOOST_ASSERT(upstreamIo.inCallHistory[1].prefetch == true);
  BOOST_ASSERT(upstreamIo.inCompleteCallHistory[0].id == tensorId);
  BOOST_ASSERT(upstreamIo.inCompleteCallHistory[0].numElements == tensorNelms);
  BOOST_ASSERT(upstreamIo.inCompleteCallHistory[1].id == tensorId);
  BOOST_ASSERT(upstreamIo.inCompleteCallHistory[1].numElements == tensorNelms);

  // STEP 4: Do another downstream fetch for replica 1 but see it doesn't result
  // in an upstream call because we didn't call inComplete.

  // Clear history between test steps.
  upstreamIo.clearHistory();
  // Request prefetch for replica 1.
  result = downstreamIos[1]->in(tensorId, tensorNelms, /*prefetch=*/true);
  // Check that the downstream result indicated success (as per upstream return
  // value).
  BOOST_ASSERT(result.data == reinterpret_cast<void *>(&inDataBuffer[7]));
  // Check that the upstream IStepIO was called twice, once with prefetch=false
  // and once with prefetch=true;
  BOOST_ASSERT(upstreamIo.inCallHistory.size() == 0);
  BOOST_ASSERT(upstreamIo.inCompleteCallHistory.size() == 0);
  BOOST_ASSERT(upstreamIo.outCallHistory.size() == 0);
  BOOST_ASSERT(upstreamIo.outCompleteCallHistory.size() == 0);

  // STEP 5: Now call inComplete on replica 1 and request new data, without
  // prefetch. Request prefetch for replica 1. Expecting fresh calls for replica
  // 3, 0 and 1.

  // Clear history between test steps.
  upstreamIo.clearHistory();
  // Tell upstreamIo what to return when called.
  upstreamIo.inReturnValues.push_back(
      ConstVoidData{reinterpret_cast<void *>(&inDataBuffer[11]), tensorInfo});
  upstreamIo.inReturnValues.push_back(
      ConstVoidData{reinterpret_cast<void *>(&inDataBuffer[16]), tensorInfo});
  upstreamIo.inReturnValues.push_back(
      ConstVoidData{reinterpret_cast<void *>(&inDataBuffer[17]), tensorInfo});
  // Do inComplete and request new data without prefetch for replica 1.
  downstreamIos[1]->inComplete(tensorId, tensorNelms);
  result = downstreamIos[1]->in(tensorId, tensorNelms, /*prefetch=*/false);
  // Check that the downstream result indicated success (as per upstream return
  // value).
  BOOST_ASSERT(result.data == reinterpret_cast<void *>(&inDataBuffer[17]));
  // Check that the upstream IStepIO was called twice, once with prefetch=false
  // and once with prefetch=true;
  BOOST_ASSERT(upstreamIo.inCallHistory.size() == 3);
  BOOST_ASSERT(upstreamIo.inCompleteCallHistory.size() == 3);
  BOOST_ASSERT(upstreamIo.outCallHistory.size() == 0);
  BOOST_ASSERT(upstreamIo.outCompleteCallHistory.size() == 0);
  BOOST_ASSERT(upstreamIo.inCallHistory[0].id == tensorId);
  BOOST_ASSERT(upstreamIo.inCallHistory[0].numElements == tensorNelms);
  BOOST_ASSERT(upstreamIo.inCallHistory[0].prefetch == false);
  BOOST_ASSERT(upstreamIo.inCallHistory[1].id == tensorId);
  BOOST_ASSERT(upstreamIo.inCallHistory[1].numElements == tensorNelms);
  BOOST_ASSERT(upstreamIo.inCallHistory[1].prefetch == false);
  BOOST_ASSERT(upstreamIo.inCallHistory[2].id == tensorId);
  BOOST_ASSERT(upstreamIo.inCallHistory[2].numElements == tensorNelms);
  BOOST_ASSERT(upstreamIo.inCallHistory[2].prefetch == false);
  BOOST_ASSERT(upstreamIo.inCompleteCallHistory[0].id == tensorId);
  BOOST_ASSERT(upstreamIo.inCompleteCallHistory[0].numElements == tensorNelms);
  BOOST_ASSERT(upstreamIo.inCompleteCallHistory[1].id == tensorId);
  BOOST_ASSERT(upstreamIo.inCompleteCallHistory[1].numElements == tensorNelms);
  BOOST_ASSERT(upstreamIo.inCompleteCallHistory[2].id == tensorId);
  BOOST_ASSERT(upstreamIo.inCompleteCallHistory[2].numElements == tensorNelms);
}

} // namespace
