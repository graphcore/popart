# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
"""
A script for extracting summarized commit messages.
"""

import subprocess

# extract commits from the top until this one is hit.
# Currently hash is (approximately) the final commit of SDK 1.2
endHsh = "325ef440b0750426b775cbc237808496e9b4f91a"

# call git log, create a giant string from it
output = (
    subprocess.Popen(["git", "log"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    .communicate()[0]
    .decode("utf-8")
)

# split the giant string into a list, one list element per commit
commits = output.split("\n\ncommit")

# walk commits until endHsh is hit, or until there are no more commits
for commit in commits:
    hsh, rest = commit.split("Author:")
    author, rest = rest.split("Date:")

    if endHsh in hsh:
        break

    # not interested in Release Agent's commits
    if "Release" in author:
        continue

    hsh = hsh.replace("commit", "").strip()
    author = author.strip().split("<")[0].strip()

    # process the commit message to extract only the commit title:
    rest = rest.split("\n")
    counter = 1
    stillSummary = True
    summaryStrings = []
    while stillSummary and counter < len(rest):
        l = rest[counter].strip()
        if "Summary:" in l or "Test Plan" in l or "Reviewers" in l:
            stillSummary = False
        elif l:
            summaryStrings.append(l)
        counter += 1
    summaryString = " ".join(summaryStrings)

    print(author)
    print(hsh)
    print(summaryString)
    print("\n")
