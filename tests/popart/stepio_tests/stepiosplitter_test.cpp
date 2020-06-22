// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE StepIOTest

#include <boost/test/unit_test.hpp>
#include <popart/stepiosplitter.hpp>

using namespace popart;

namespace {

// Test class that allows us to feed in a custom stream of in/out data objects.
class TestStepIO : public IStepIO {
public:
  TestStepIO(const std::list<ConstVoidData> &testInData_,
             const std::list<MutableVoidData> &testOutData_)
      : testInData(testInData_), testOutData(testOutData_) {}

  virtual ~TestStepIO() = default;

  virtual ConstVoidData in(TensorId id, int64_t numElements, bool prefetch) {
    BOOST_ASSERT(!testInData.empty());
    return testInData.front();
  }

  virtual void inComplete(TensorId id, int64_t numElements) {
    return testInData.pop_front();
  }

  virtual MutableVoidData out(TensorId id, int64_t numElements) {
    BOOST_ASSERT(!testOutData.empty());
    return testOutData.front();
  }

  virtual void outComplete(TensorId) { return testOutData.pop_front(); }

  virtual void assertNumElements(const Ir &) const {
    // pass
  }

  std::list<ConstVoidData> testInData;
  std::list<MutableVoidData> testOutData;
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

  // Now try a slightly more chaotic order of downstream calls.
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

BOOST_AUTO_TEST_CASE(StepIOSplitter_Test) {

  const size_t NUM_INPUTS     = 100;
  const unsigned NUM_REPLICAS = 4;

  TensorId tensorId{"testTensor1"};
  TensorInfo tensorInfo{DataType::FLOAT, Shape{1}};
  int64_t tensorNelms{tensorInfo.nelms()};

  // Data buffer we can use for comparing memory locations.
  std::vector<unsigned> inDataBuffer{NUM_INPUTS, 0u};
  std::vector<unsigned> outDataBuffer{NUM_INPUTS, 0u};

  // Set up upstream IStepIO.
  std::list<ConstVoidData> upstreamTestDataIn;
  std::list<MutableVoidData> upstreamTestDataOut;

  for (int i = 0; i < NUM_INPUTS; ++i) {
    ConstVoidData inData;
    inData.data = static_cast<void *>(&inDataBuffer[i]);
    inData.info = tensorInfo;
    upstreamTestDataIn.push_back(inData);

    MutableVoidData outData;
    outData.data = static_cast<void *>(&outDataBuffer[i]);
    outData.info = tensorInfo;
    upstreamTestDataOut.push_back(outData);
  }

  TestStepIO upstreamIo(upstreamTestDataIn, upstreamTestDataOut);

  // Set up StepIOSplitter.
  StepIOSplitter splitter(NUM_REPLICAS);
  splitter.reset(&upstreamIo);

  // Get downstream StepIOs from splitter.
  std::vector<IStepIO *> downstreamIos;
  for (unsigned replicationIndex = 0; replicationIndex < NUM_REPLICAS;
       ++replicationIndex) {
    downstreamIos.push_back(
        splitter.getDownstreamStepIO(tensorId, replicationIndex));
  }

  // Get downstream Ios again for some replicas to ensure it won't cause
  // problems.
  splitter.getDownstreamStepIO(tensorId, 0);
  splitter.getDownstreamStepIO(tensorId, 3);

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

} // namespace
