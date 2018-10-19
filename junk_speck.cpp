// The consumer which is best
// for creating the poplar::Tensor
// The consuming ops vote on what unique

// Speck the consumed tensor should have.
// Rules for voting on the Speck:
// 1) if all consumers expect Speck X,
//    return X, where X could be Any, ConvWeight, etc.
// 2) if all consumers expect either Any or X,
//    return X
// 3) if number of different Specks of consumers
//    is greater than 2, error.
OpId getPriviledgedConsumer(Tensor *);

// The (Spec)ific "type" of a tensor,
// as expected by a consumer of the tensor
enum class Speck {
  ConvWeight = 0,
  ConvBias,
  ConvData,
  Any,
  N // number of tensor specks
};


const std::map<Speck, SpeckInfo> &getSpeckMap() {
  static std::map<Speck, SpeckInfo> M = initSpeckMap();
  return M;
}


Speck Consumers::consensusSpeck() {
  std::vector<Speck> allSpecks;
  // for all consumers of this tensor,
  for (Op *consumer : getOps()) {
    // and for all indices at which it is consumed,
    auto indices = consumer->input.indices(tensorConsumed);
    for (int index : indices) {
      // what is the Speck at this index?
      // i.e. what kind of Tensor is expected at this index?
      auto speck = consumer->inputSpeckAt(index);
      if (std::find(allSpecks.begin(), allSpecks.end(), speck) ==
          allSpecks.end()) {
        allSpecks.push_back(speck);
      }
    }
  }

  // Rule 1)
  if (allSpecks.size() == 1) {
    return allSpecks[0];
  }

  // Rule 2)
  else if (allSpecks.size() == 2) {
    if (allSpecks[0] == Speck::Any) {
      return allSpecks[1];
    } else if (allSpecks[1] == Speck::Any) {
      return allSpecks[0];
    }
  }

  // Rule 3)
  std::stringstream errm;
  errm << "Failed to determine Speck for " << tensorConsumed->id
       << ", consumers expected : ";
  for (auto &speck : allSpecks) {
    errm << getSpeckMap().at(speck).speck_s();
    errm << " ";
  }
  throw error(errm.str());
}

SpeckInfo::SpeckInfo(Speck s_, std::string s_s_) : speck_(s_), speck_s_(s_s_) {}

std::map<Speck, SpeckInfo> initSpeckMap() {
  std::map<Speck, SpeckInfo> specks_m = {
      {Speck::ConvWeight, {Speck::ConvWeight, "ConvWeight"}},
      {Speck::ConvBias, {Speck::ConvBias, "ConvBias"}},
      {Speck::ConvData, {Speck::ConvData, "ConvData"}},
      {Speck::Any, {Speck::Any, "Any"}}};
  if (specks_m.size() != static_cast<int64_t>(Speck::N)) {
    throw error("missing element in Specks");
  }
  return specks_m;
}


Speck Op::inputSpeckAt(int index) const {
  // perform a health check: is the index a valid input index?
  if (!input.hasIndex(index)) {
    throw error("no input index " + std::to_string(index));
  }
  // this is the default return type for an Op, this will be overwritten
  // where specific Specks are needed.
  return Speck::Any;
}


class SpeckInfo {
public:
  SpeckInfo(Speck, std::string);
  Speck speck() const;
  const std::string &speck_s() const;

private:
  Speck speck_;
  std::string speck_s_;
};
const std::map<Speck, SpeckInfo> &getSpeckMap();
std::map<Speck, SpeckInfo> initSpeckMap();

Speck SpeckInfo::speck() const { return speck_; }

// Conv has specialised input Speck, we override the default here
Speck ConvOp::inputSpeckAt(int index) const {
  static const std::map<int, Speck> M = createSpeckMap();
  return M.at(index);
}


std::map<int, Speck> ConvOp::createSpeckMap() const {
  std::map<int, Speck> M;
  M[weightsInIndex()] = Speck::ConvWeight;
  M[dataInIndex()]    = Speck::ConvData;
  return M;
}


  Speck inputSpeckAt(int index) const override final;
  std::map<int, Speck> createSpeckMap() const;


  // The specific tensor "Speck" expected at input index
  // default return : Speck::Any. That is, if not specified,
  // assume any Speck is valid
  virtual Speck inputSpeckAt(int) const;



const std::string &SpeckInfo::speck_s() const { return speck_s_; }


//    Speck speck        = initTensor->consumers.consensusSpeck();
//    std::cout << getSpeckMap().at(speck).speck_s() << std::endl;
//    switch (speck) {
//    case Speck::ConvWeight: {
//      throw error("Cannot create tensor for conv weight yet");
//    }
//    case Speck::ConvBias: {
//      throw error("Cannot create tensor for conv bias yet");
//    }
//    case Speck::ConvData: {
//      throw error("Cannot create tensor for conv data yet");
//    }
//    case Speck::Any: {
//      throw error("Cannot create tensor for Any yet");
//    }
//    case Speck::N: {
//      throw error("ILE : Speck::N is not a true Speck");
//    }
//    }


//    Speck speck        = initTensor->consumers.consensusSpeck();
//    std::cout << getSpeckMap().at(speck).speck_s() << std::endl;
//    switch (speck) {
//    case Speck::ConvWeight: {
//      throw error("Cannot create tensor for conv weight yet");
//    }
//    case Speck::ConvBias: {
//      throw error("Cannot create tensor for conv bias yet");
//    }
//    case Speck::ConvData: {
//      throw error("Cannot create tensor for conv data yet");
//    }
//    case Speck::Any: {
//      throw error("Cannot create tensor for Any yet");
//    }
//    case Speck::N: {
//      throw error("ILE : Speck::N is not a true Speck");
//    }
//    }

//    Speck speck        = initTensor->consumers.consensusSpeck();
//    std::cout << getSpeckMap().at(speck).speck_s() << std::endl;
//    switch (speck) {
//    case Speck::ConvWeight: {
//      throw error("Cannot create tensor for conv weight yet");
//    }
//    case Speck::ConvBias: {
//      throw error("Cannot create tensor for conv bias yet");
//    }
//    case Speck::ConvData: {
//      throw error("Cannot create tensor for conv data yet");
//    }
//    case Speck::Any: {
//      throw error("Cannot create tensor for Any yet");
//    }
//    case Speck::N: {
//      throw error("ILE : Speck::N is not a true Speck");
//    }
//    }
