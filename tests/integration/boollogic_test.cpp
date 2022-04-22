// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE BoolLogicTest

#include <boost/test/unit_test.hpp>
#include <map>
#include <string>
#include <popart/boollogic.hpp>

using namespace popart;
using namespace boollogic;

BOOST_AUTO_TEST_CASE(Term_Not0) {
  Term term0 = Term::notTerm(Term::notTerm(Term::varTerm("A")));

  Term term1 = term0.pushNots();

  BOOST_CHECK_EQUAL(term1, Term::varTerm("A"));
  BOOST_CHECK_NE(term1, term0);
  BOOST_CHECK_NE(term1, Term::varTerm("B"));
}

BOOST_AUTO_TEST_CASE(Term_Not1) {
  Term term0 =
      Term::notTerm(Term::andTerm({Term::varTerm("A"),
                                   Term::varTerm("B"),
                                   Term::notTerm(Term::varTerm("C"))}));
  Term term1 = Term::notTerm(Term::orTerm({Term::varTerm("A"),
                                           Term::notTerm(Term::varTerm("B")),
                                           Term::varTerm("C")}));
  Term term2 = Term::orTerm({Term::notTerm(Term::varTerm("A")),
                             Term::notTerm(Term::varTerm("B")),
                             Term::varTerm("C")});
  Term term3 = Term::andTerm({Term::notTerm(Term::varTerm("A")),
                              Term::varTerm("B"),
                              Term::notTerm(Term::varTerm("C"))});

  Term term4 = term0.pushNots();
  Term term5 = term1.pushNots();
  Term term6 = term2.pushNots();
  Term term7 = term3.pushNots();

  BOOST_CHECK_EQUAL(term4, term2);
  BOOST_CHECK_EQUAL(term5, term3);
  BOOST_CHECK_EQUAL(term6, term2);
  BOOST_CHECK_EQUAL(term7, term3);
}

BOOST_AUTO_TEST_CASE(Term_Not2) {
  Term term0 = Term::notTerm(Term::andTerm(
      {Term::varTerm("A"),
       Term::orTerm({Term::varTerm("B"), Term::notTerm(Term::varTerm("C"))})}));
  Term term1 = Term::orTerm(
      {Term::notTerm(Term::varTerm("A")),
       Term::andTerm({Term::notTerm(Term::varTerm("B")), Term::varTerm("C")})});

  Term term2 = term0.pushNots();

  std::map<std::string, bool> vals;
  vals["A"] = false;
  vals["B"] = true;
  vals["C"] = false;

  BOOST_CHECK_EQUAL(term1, term2);
  BOOST_CHECK(term0.evaluate(vals));
  BOOST_CHECK(term1.evaluate(vals));

  vals["A"] = true;
  vals["B"] = true;
  vals["C"] = false;

  BOOST_CHECK(!term0.evaluate(vals));
  BOOST_CHECK(!term1.evaluate(vals));
}

BOOST_AUTO_TEST_CASE(Term_Flatten0) {
  Term a = Term::varTerm("A");
  Term b = Term::varTerm("B");
  Term c = Term::varTerm("C");
  Term d = Term::varTerm("D");

  Term t0 = (a || ((b && c) && d));
  Term t1 = (a && ((b || c) || d));

  Term t2 = (a || Term::andTerm({b, c, d}));
  Term t3 = (a && Term::orTerm({b, c, d}));

  Term t4 = t0.flatten();
  Term t5 = t1.flatten();

  BOOST_CHECK_EQUAL(t4, t2);
  BOOST_CHECK_EQUAL(t5, t3);
}

BOOST_AUTO_TEST_CASE(Term_Insert0) {
  Term a = Term::varTerm("A");
  Term b = Term::varTerm("B");
  Term c = Term::varTerm("C");
  Term d = Term::varTerm("D");

  Term t0 = (a || ((b && c) || d));
  Term t1 = (a && ((b || c) && d));

  Term t2 = (a || ((b && a) || b));
  Term t3 = (a && ((b || a) && b));

  std::map<std::string, Term> iterms;
  iterms.insert({"C", a});
  iterms.insert({"D", b});

  Term t4 = t0.replace(iterms);
  Term t5 = t1.replace(iterms);

  BOOST_CHECK_EQUAL(t2, t4);
  BOOST_CHECK_EQUAL(t3, t5);
}

BOOST_AUTO_TEST_CASE(Term_CNF0) {
  Term a = Term::varTerm("A");
  Term b = Term::varTerm("B");
  Term c = Term::varTerm("C");
  Term d = Term::varTerm("D");
  Term e = Term::varTerm("E");

  Term t0 = ((a && b) || (c && d)) || e;
  Term t1 = t0.getCNF();
  Term t2 = ((c || e || a) && (c || e || b) && (d || e || a) && (d || e || b))
                .flatten();

  BOOST_CHECK_EQUAL(t1, t2);
}

BOOST_AUTO_TEST_CASE(Term_DNF0) {
  Term a = Term::varTerm("A");
  Term b = Term::varTerm("B");
  Term c = Term::varTerm("C");
  Term d = Term::varTerm("D");
  Term e = Term::varTerm("E");

  Term t0 = ((a || b) && (c || d)) && e;
  Term t1 = t0.getDNF();
  Term t2 = ((c && e && a) || (c && e && b) || (d && e && a) || (d && e && b))
                .flatten();

  BOOST_CHECK_EQUAL(t1, t2);
}
