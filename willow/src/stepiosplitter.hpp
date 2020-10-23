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
class SplitIOTensorInfo;

// A helper class that acts as a downstream interface for input and output data
// streams.
class StepIOSplitterAdapter : public IStepIO {
public:
  // Constructor.
  StepIOSplitterAdapter(StepIOSplitter *splitter,
                        SplitIOTensorInfo *tensorInfo,
                        unsigned replicationIndex,
                        unsigned replicationFactor,
                        TensorId id,
                        const TensorInfo &info,
                        const int maxInFetches,
                        const int maxOutFetches);
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

  // Log the state (for in).
  void inLog(const char *action) const;
  // Log the state (for out).
  void outLog(const char *action) const;

  // Reset all in/out data.
  void reset();

  // Can we add data to the in buffer?
  bool canAddInBuffer() const;
  // Can we add data to the out buffer?
  bool canAddOutBuffer() const;

  // Add inData buffer.
  void addInBuffer(const ConstVoidData &buf);
  // Add outData buffer.
  void addOutBuffer(const MutableVoidData &buf);

  // See if there's an input buffer ready to complete upstream. If so,
  // update the bookkeeping as if this inComplete has happened.
  bool tryInCompleteUpstream();
  // See if there's an output buffer ready to complete upstream. If so,
  // update the bookkeeping as if this outComplete has happened.
  bool tryOutCompleteUpstream();

private:
  // Reference back to StepIOSplitter object.
  StepIOSplitter *splitter;
  // Reference back to SplitIOTensorInfo object.
  SplitIOTensorInfo *tensorInfo;
  // Replication index.
  unsigned replicationIndex;
  // Number of replicas.
  unsigned replicationFactor;
  // The id this adapter was created for.
  TensorId adapterId;
  // Maximum number of in fetches.
  int maxInFetches;
  // Maximum number of out fetches.
  int maxOutFetches;
  // Buffer of elements to ready to read.
  std::list<ConstVoidData> inData;
  // Number of input fetches.
  unsigned numInFetches;
  // Number of in buffers yet to be completed by poplar.
  unsigned numInIncompleteDownstream;
  // Number of in buffers completed by poplar but still incomplete because of
  // order.
  unsigned numInIncompleteUpstream;
  // Buffer of elements to ready to write.
  std::list<MutableVoidData> outData;
  // Number of output fetches.
  unsigned numOutFetches;
  // Number of out buffers yet to be completed by poplar.
  unsigned numOutIncompleteDownstream;
  // Number of out buffers completed by poplar but still incomplete because of
  // order.
  unsigned numOutIncompleteUpstream;
  // Void data to return if input with prefetch fails.
  ConstVoidData emptyVoidData;
};

// Helper class to store state for every tensor.
class SplitIOTensorInfo {
public:
  // Default constructor.
  SplitIOTensorInfo();

  // The replica index that is next in line to receive 'in' data.
  unsigned inIndex;
  unsigned inCompleteIndex;
  // The replica index that is next in line to receive 'out' data.
  unsigned outIndex;
  unsigned outCompleteIndex;

  // Map from replication indices to IStepIO adapters
  std::map<unsigned, std::unique_ptr<StepIOSplitterAdapter>> adapterMap;
};

// A class that splits one StepIO interface into multiple StepIO interfaces that
// can be read/written to by multiple replicas separately.
class StepIOSplitter {
public:
  // Constructor.
  // \param replicationFactor the number of replicas.
  // \param maxInFetchesPerReplFun a function mapping tensor ids to the maximum
  //     number of input buffer fetches per replica expected for this tensor.
  // \param maxOutFetchesPerReplFun a function mapping tensor ids to the maximum
  //     number of output buffer fetches per replica expected for this tensor.
  StepIOSplitter(
      unsigned replicationFactor,
      std::function<unsigned(const TensorId &)> maxInFetchesPerReplFun,
      std::function<unsigned(const TensorId &)> maxOutFetchesPerReplFun);
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

  // Give the splitter a change to call inComplete upstream.
  virtual void inCompletionCallback(TensorId id,
                                    int64_t numElements,
                                    unsigned replicationIndex);
  // Give the splitter a change to call outComplete upstream.
  virtual void outCompletionCallback(TensorId id, unsigned replicationIndex);

private:
  // The number of replications.
  unsigned replicationFactor;
  // Function to get maximum input fetches.
  std::function<unsigned(const TensorId &)> maxInFetchesPerReplFun;
  // Function to get maximum output fetches.
  std::function<unsigned(const TensorId &)> maxOutFetchesPerReplFun;

  // The upstream datastream.
  IStepIO *upstreamIo;
  // Map tuples TensorId to a map from replication indices to IStepIO adapters.
  std::map<TensorId, SplitIOTensorInfo> downstreamIoMap;
};

} // namespace popart

#endif
