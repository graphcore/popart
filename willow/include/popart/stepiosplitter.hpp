// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_STEPIOSPLITTER_HPP
#define GUARD_NEURALNET_STEPIOSPLITTER_HPP

#include <popart/istepio.hpp>

#include <list>
#include <map>
#include <tuple>

namespace popart {

// Forward declaration.
class StepIOSplitter;

// A helper class that acts as a downstream interface for input and output data
// streams.
class StepIOSplitterAdapter : public IStepIO {
public:
  // Constructor.
  StepIOSplitterAdapter(StepIOSplitter *splitter,
                        unsigned replicationIndex,
                        TensorId id,
                        const TensorInfo &info);
  // Destructor.
  virtual ~StepIOSplitterAdapter() = default;
  // Get next data element for reading from adapter.
  virtual ConstVoidData in(TensorId id, int64_t numElements, bool prefetch);
  // Move on to next data element.
  virtual void inComplete(TensorId id, int64_t numElements);
  // Get next data element for writing from adapter.
  virtual MutableVoidData out(TensorId id, int64_t numElements);
  // Move on to next data element.
  virtual void outComplete(TensorId);
  // Check number of elements.
  virtual void assertNumElements(const Ir &ir) const;

  // Reset all in/out data.
  void reset();

  // Get reference to in data buffer.
  std::list<ConstVoidData> &getInData() { return inData; }
  // Get reference to out data buffer.
  std::list<MutableVoidData> &getOutData() { return outData; }

private:
  // Reference back to StepIOSplitter object.
  StepIOSplitter *splitter;
  // Replication index.
  unsigned replicationIndex;
  // The id this adapter was created for.
  TensorId adapterId;
  // Buffer of elements to read from.
  std::list<ConstVoidData> inData;
  // Buffer of elements to write into.
  std::list<MutableVoidData> outData;
  // Void data to return if input with prefetch fails.
  ConstVoidData emptyVoidData;
};

// Helper class to store state for every tensor.
class SplitIOTensorInfo {
public:
  // Default constructor.
  SplitIOTensorInfo();

  // The number of data elements loaded so far
  unsigned fetchCount;
  // The replica index that is next in line to receive 'in' data.
  unsigned inIndex;
  // The replica index that is next in line to receive 'out' data.
  unsigned outIndex;
  // True if a call to inComplete is pending.
  bool upstreamInCompletePending;

  // Map from replication indices to IStepIO adapters
  std::map<unsigned, std::unique_ptr<StepIOSplitterAdapter>> adapterMap;
};

// A class that splits one StepIO interface into multiple StepIO interfaces that
// can be read/written to by multiple replicas separately.
class StepIOSplitter {
public:
  // Constructor.
  StepIOSplitter(unsigned replicationFactor,
                 unsigned batchesPerStep,
                 unsigned accumulationFactor);
  // Don't allow copying.
  StepIOSplitter(const StepIOSplitter &) = delete;
  // Don't allow assigning.
  StepIOSplitter &operator=(const StepIOSplitter &) = delete;
  // Destructor.
  virtual ~StepIOSplitter() = default;

  // Reset the logic.
  void reset();
  // Set the upstream IStepIO.
  void setUpstreamIo(IStepIO *upstreamIo);

  // Fetch data from upstream for a specific replica (getting data for preceding
  // replicas first, if necessary, to avoid calling the upstream IStepIO out of
  // order).
  void getInData(TensorId id,
                 int64_t numElements,
                 unsigned replicationIndex,
                 bool prefetch);
  // Fetch output buffer from upstream for a specific replica (getting data for
  // preceding replicas first, if necessary, to avoid calling the upstream
  // IStepIO out of order).
  void getOutData(TensorId id, int64_t numElements, unsigned replicationIndex);

  // Check number of elements in upstream IStepIO.
  virtual void assertNumElements(const Ir &) const;

  // Get access to the 'split' data stream.
  IStepIO *getDownstreamStepIO(TensorId id,
                               const TensorInfo &info,
                               unsigned replicationIndex);

  // Give the splitter a change to call inComplete upstream from a downstream
  // inComplete call.
  virtual void inCompleteCallback(TensorId id,
                                  int64_t numElements,
                                  unsigned replicationIndex);

private:
  // The number of replications.
  unsigned replicationFactor;

  // The number of batches per step.
  unsigned batchesPerStep;

  // The accumulation factor.
  unsigned accumulationFactor;

  // The upstream datastream.
  IStepIO *upstreamIo;
  // Map tuples TensorId to a map from replication indices to IStepIO adapters.
  std::map<TensorId, SplitIOTensorInfo> downstreamIoMap;
};

} // namespace popart

#endif
