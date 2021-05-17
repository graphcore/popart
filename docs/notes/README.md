# Developer Notes

This directory is intended to be used what we call *Developer notes*. These are
documentation items that are directed at PopART developers, as opposed to users,
and therefore do not belong in neither the API documentation or the User Guide.

## Rules

* Don't document things here that belong in the API documentation or user guide
  or that are better placed near code in comments.
* Make sure the documentation is *maintainable*. That is, if you are using a
  tool to produce a diagram, make sure it's possible to update said diagrams and
  provide instructions that describe how to accomplish this. Avoid using tools
  that may not be available to all team members (or the use of which goes
  against company policy).

  Some diagrams in some documents were created with [draw.io](https://draw.io/).
  To edit these diagrams, edit the relevant `.drawio` file via
  [draw.io](https://draw.io/) and then both save the `.drawio` file and export a
  new `.png` file.
* Generally speaking follow the same
  organisation as the `willow/include` directory.
* Follow Graphcore's documentation guidelines (see G.MAP).
