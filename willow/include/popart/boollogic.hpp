// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_BOOLLOGIC_HPP
#define GUARD_NEURALNET_BOOLLOGIC_HPP

#include <iosfwd>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace popart {
namespace boollogic {

enum class Type { Not = 0, And, Or, Var, True, False };

std::ostream &operator<<(std::ostream &os, const Type &type);

class Term {
public:
  Term(std::string var);
  Term(bool val);

  static Term trueTerm();
  static Term falseTerm();
  static Term varTerm(const std::string &var);
  static Term notTerm(Term term);
  static Term andTerm(const std::set<Term> &terms);
  static Term orTerm(const std::set<Term> &terms);
  static Term andTermFromVector(const std::vector<Term> &terms);
  static Term orTermFromVector(const std::vector<Term> &terms);

  Type getType() const { return type; }

  const std::set<Term> &getTerms() const { return terms; }
  std::vector<Term> getTermsAsVector() const;

  std::string getVar() const { return var; }

  // Pushes nots down using demorgan's law
  Term pushNots() const;

  // Convert nested And/Or terms into variadic terms where possible
  Term flatten() const;

  // Returns the conjunctive normal form of the term
  Term getCNF() const;

  // Returns the disjunctive normal form of the term
  Term getDNF() const;

  // Applies distributive law over And or Or until no longer possible
  Term distribute(Type dtype) const;

  Term operator&&(const Term &other) const;
  Term operator||(const Term &other) const;
  Term operator!() const;
  bool operator==(const Term &other) const;
  bool operator<(const Term &other) const;

  std::string str() const;

  Term replace(const std::map<std::string, Term> &terms) const;
  bool evaluate(const std::map<std::string, bool> &vals) const;

private:
  Term pushNots(bool isNot) const;

  Term(Type type, std::string var, std::set<Term> terms);

  Type type;
  std::string var;
  std::set<Term> terms;
};

std::ostream &operator<<(std::ostream &os, const Term &term);

} // namespace boollogic
} // namespace popart
#endif
