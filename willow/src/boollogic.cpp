// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <algorithm>
#include <map>
#include <ostream>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include <popart/boollogic.hpp>
#include <popart/error.hpp>
#include <popart/logging.hpp>

namespace popart {
namespace boollogic {

Term::Term(Type type_, std::string var_, std::set<Term> terms_)
    : type(type_), var(var_), terms() {
  terms.insert(terms_.begin(), terms_.end());
}

Term::Term(std::string var_) : type(Type::Var), var(var_), terms() {}
Term::Term(bool val) : type(val ? Type::True : Type::False), var(), terms() {}

Term Term::trueTerm() { return Term(Type::True, "", {}); }

Term Term::falseTerm() { return Term(Type::False, "", {}); }

Term Term::varTerm(const std::string &var) { return Term(Type::Var, var, {}); }

Term Term::notTerm(Term term) { return Term(Type::Not, "", {term}); }

Term Term::andTerm(const std::set<Term> &terms) {
  if (terms.empty()) {
    return Term::trueTerm();
  }

  return Term(Type::And, "", terms);
}

Term Term::orTerm(const std::set<Term> &terms) {
  if (terms.empty()) {
    return Term::falseTerm();
  }
  return Term(Type::Or, "", terms);
}

Term Term::andTermFromVector(const std::vector<Term> &terms) {
  if (terms.empty()) {
    return Term::trueTerm();
  }
  std::set<Term> termSet(terms.begin(), terms.end());
  return Term(Type::And, "", termSet);
}

Term Term::orTermFromVector(const std::vector<Term> &terms) {
  if (terms.empty()) {
    return Term::falseTerm();
  }
  std::set<Term> termSet(terms.begin(), terms.end());
  return Term(Type::Or, "", termSet);
}

std::vector<Term> Term::getTermsAsVector() const {
  std::vector<Term> vterms;
  vterms.reserve(terms.size());
  vterms.insert(vterms.begin(), terms.begin(), terms.end());
  return vterms;
}

Term Term::pushNots() const { return pushNots(false); }

Term Term::pushNots(bool isNot) const {
  switch (type) {
  case Type::Not: {
    if (terms.size() != 1) {
      throw error("Unsupported term {} with {} subterms", type, terms.size());
    }
    return (*(terms.begin())).pushNots(!isNot);
  }
  case Type::And: {
    std::set<Term> subterms;
    for (auto &t : terms) {
      subterms.insert(t.pushNots(isNot));
    }
    return isNot ? orTerm(subterms) : andTerm(subterms);
  }
  case Type::Or: {
    std::set<Term> subterms;
    for (auto &t : terms) {
      subterms.insert(t.pushNots(isNot));
    }
    return isNot ? andTerm(subterms) : orTerm(subterms);
  }
  case Type::Var: {
    return isNot ? notTerm(*this) : *this;
  }
  case Type::True: {
    return isNot ? falseTerm() : *this;
  }
  case Type::False: {
    return isNot ? trueTerm() : *this;
  }
  default:
    throw error("Unsupported term {}", static_cast<int>(type));
  }
}

Term Term::operator&&(const Term &other) const {
  return andTerm({*this, other});
}

Term Term::operator||(const Term &other) const {
  return orTerm({*this, other});
}

Term Term::operator!() const { return notTerm(*this); }

bool Term::operator==(const Term &other) const {
  if (type != other.type) {
    return false;
  }
  if (terms.size() != other.terms.size()) {
    return false;
  }
  if (var != other.var) {
    return false;
  }
  return terms == other.terms;
}

bool Term::operator<(const Term &other) const {
  return std::make_tuple<const Type &,
                         const std::string &,
                         const std::set<Term> &>(type, var, terms) <
         std::make_tuple<const Type &,
                         const std::string &,
                         const std::set<Term> &>(
             other.type, other.var, other.terms);
}

std::string Term::str() const {
  switch (type) {
  case Type::Not: {
    if (terms.size() != 1) {
      throw error("Unsupported term {} with {} subterms", type, terms.size());
    }
    return "!" + (*terms.begin()).str();
  }
  case Type::And: {
    std::vector<std::string> subterms;
    subterms.reserve(terms.size());
    for (auto &t : terms) {
      subterms.push_back(t.str());
    }
    return "(" + logging::join(subterms.begin(), subterms.end(), " && ") + ")";
  }
  case Type::Or: {
    std::vector<std::string> subterms;
    subterms.reserve(terms.size());
    for (auto &t : terms) {
      subterms.push_back(t.str());
    }
    return "(" + logging::join(subterms.begin(), subterms.end(), " || ") + ")";
  }
  case Type::Var: {
    return var;
  }
  case Type::True: {
    return "true";
  }
  case Type::False: {
    return "false";
  }
  default:
    throw error("Unsupported term {}", type);
  }
}

bool Term::evaluate(const std::map<std::string, bool> &vals) const {
  switch (type) {
  case Type::Not: {
    if (terms.size() != 1) {
      throw error("Unsupported term {} with {} subterms", type, terms.size());
    }
    return !(*terms.begin()).evaluate(vals);
  }
  case Type::And: {
    return std::all_of(terms.begin(), terms.end(), [&vals](const Term &t) {
      return t.evaluate(vals);
    });
  }
  case Type::Or: {
    return std::any_of(terms.begin(), terms.end(), [&vals](const Term &t) {
      return t.evaluate(vals);
    });
  }
  case Type::Var: {
    auto it = vals.find(var);
    if (it != vals.end()) {
      return it->second;
    }
    throw error("Variable {} not defined", var);
  }
  case Type::True: {
    return true;
  }
  case Type::False: {
    return false;
  }
  default:
    throw error("Unsupported term {}", type);
  }
}

Term Term::replace(const std::map<std::string, Term> &iterms) const {
  switch (type) {
  case Type::Not: {
    if (terms.size() != 1) {
      throw error("Unsupported term {} with {} subterms", type, terms.size());
    }
    return !(*terms.begin()).replace(iterms);
  }
  case Type::And: {
    std::set<Term> subterms;
    for (auto &t : terms) {
      subterms.insert(t.replace(iterms));
    }
    return andTerm(subterms);
  }
  case Type::Or: {
    std::set<Term> subterms;
    for (auto &t : terms) {
      subterms.insert(t.replace(iterms));
    }
    return orTerm(subterms);
  }
  case Type::Var: {
    auto it = iterms.find(var);
    if (it != iterms.end()) {
      return it->second;
    }
    return *this;
  }
  case Type::True: {
    return *this;
  }
  case Type::False: {
    return *this;
  }
  default:
    throw error("Unsupported term {}", type);
  }
}

Term Term::flatten() const {
  switch (type) {
  case Type::Not: {
    if (terms.size() != 1) {
      throw error("Unsupported term {} with {} subterms", type, terms.size());
    }
    return notTerm((*terms.begin()).flatten());
  }
  case Type::And: {
    std::set<Term> subterms;
    for (auto &t : terms) {
      auto tflat = t.flatten();
      if (t.type == type) {
        subterms.insert(tflat.terms.begin(), tflat.terms.end());
      } else {
        subterms.insert(tflat);
      }
    }
    return andTerm(subterms);
  }
  case Type::Or: {
    std::set<Term> subterms;
    for (auto &t : terms) {
      auto tflat = t.flatten();
      if (t.type == type) {
        subterms.insert(tflat.terms.begin(), tflat.terms.end());
      } else {
        subterms.insert(tflat);
      }
    }
    return orTerm(subterms);
  }
  case Type::Var: {
    return *this;
  }
  case Type::True: {
    return *this;
  }
  case Type::False: {
    return *this;
  }
  default:
    throw error("Unsupported term {}", type);
  }
}

Term Term::distribute(Type dtype) const {

  auto dist = [this, dtype]() {
    Type otype = dtype == Type::Or ? Type::And : Type::Or;
    if (dtype == type) {
      // Distribute term
      Term tflat               = flatten();
      std::vector<Term> vterms = tflat.getTermsAsVector();
      // Sort so that terms that can't be distributed over are processed first
      std::stable_sort(
          vterms.begin(), vterms.end(), [otype](const Term &a, const Term &b) {
            return a.type != otype && b.type == otype;
          });
      Term term0 = vterms.at(0);
      for (const Term &term1 : vterms) {
        if (term1.type == otype) {
          // Distribute
          std::set<Term> subterms;
          for (Term t : term1.terms) {
            t = dtype == Type::Or ? orTerm({term0, t}) : andTerm({term0, t});
            subterms.insert(t.distribute(dtype));
          }
          term0 = otype == Type::Or ? orTerm(subterms) : andTerm(subterms);
        } else {
          // Can't distribute
          term0 = (dtype == Type::Or ? orTerm({term0, term1})
                                     : andTerm({term0, term1}))
                      .flatten();
        }
      }
      return term0.flatten();
    } else {
      // Distribute subterms
      std::set<Term> subterms;
      for (auto &t : terms) {
        subterms.insert(t.distribute(dtype));
      }
      return type == Type::Or ? orTerm(subterms).flatten()
                              : andTerm(subterms).flatten();
    }
  };

  switch (type) {
  case Type::Not: {
    if (terms.size() != 1) {
      throw error("Unsupported term {} with {} subterms", type, terms.size());
    }
    return !(*terms.begin()).distribute(dtype);
  }
  case Type::And: {
    return dist();
  }
  case Type::Or: {
    return dist();
  }
  case Type::Var: {
    return *this;
  }
  case Type::True: {
    return *this;
  }
  case Type::False: {
    return *this;
  }
  default:
    throw error("Unsupported term {}", type);
  }
}

Term Term::getCNF() const { return pushNots().distribute(Type::Or); }

Term Term::getDNF() const { return pushNots().distribute(Type::And); }

std::ostream &operator<<(std::ostream &os, const Term &term) {
  os << term.str();
  return os;
}

std::ostream &operator<<(std::ostream &os, const Type &type) {
  switch (type) {
  case Type::Not: {
    os << "Not";
    break;
  }
  case Type::And: {
    os << "And";
    break;
  }
  case Type::Or: {
    os << "Or";
    break;
  }
  case Type::Var: {
    os << "Var";
    break;
  }
  case Type::True: {
    os << "True";
    break;
  }
  case Type::False: {
    os << "False";
    break;
  }
  }
  return os;
}

} // namespace boollogic
} // namespace popart
