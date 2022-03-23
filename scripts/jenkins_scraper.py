# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
"""
    A simple script to pull PopART build/test stats from Jenkins.

    You need Beautiful Soup installed for this to work. Use:

    ```
    > pip install beautifulsoup4==4.10.0
    ```
"""

import argparse
import collections
import io
import os
import re
import requests
import sys
import tarfile

try:
    from bs4 import BeautifulSoup
except:
    print(
        "You do not have beautiful soup installed (try: 'pip install beautifulsoup4')."
    )
    sys.exit(1)

cookie_file = os.path.join(os.path.expanduser("~"), ".artifactory_ci_cookie")


def get_cookies():
    """ Return authentication cookies if they exist. """
    if not os.path.exists(cookie_file):
        print(
            f"This script requires a cookie ({cookie_file}). See https://cookiecutter.sourcevertex.net/ for details."
        )
        sys.exit(-1)

    with open(cookie_file) as f:
        cookies_str = "".join(f.readlines())

        def get_key(cookie_str):
            cookie_str = cookie_str.strip()
            first_eq = cookie_str.find("=")
            return cookie_str[0:first_eq]

        def get_val(cookie_str):
            cookie_str = cookie_str.strip()
            first_eq = cookie_str.find("=")
            return cookie_str[first_eq + 1:]

        cookies = {
            get_key(cookie_str): get_val(cookie_str)
            for cookie_str in cookies_str.split("; ")
        }
        return cookies


def get_console_text(baseurl):
    """Get the console text for a child job as a string."""
    fetch_url = f'{baseurl}/consoleText'
    page = requests.get(fetch_url, cookies=get_cookies())
    assert page.status_code == 200
    return page.text  #page.content.decode("utf8")


def get_ninja_log(baseurl, project):
    """Get ninja log for a specific project in a child job as a string."""
    # Get link to ninja log from project page
    fetch_url = f'{baseurl}/'
    page = requests.get(fetch_url, cookies=get_cookies())
    assert page.status_code == 200
    soup = BeautifulSoup(page.text, 'html.parser')
    links = soup.find_all('a', href=True, text='Build log: ninja logs')
    assert (len(links) == 1)
    link = links[0]['href']

    # Get the tar.gz file.
    targz_request = requests.get(link, cookies=get_cookies(), stream=True)
    assert targz_request.status_code == 200

    # Wrap tarfile into memory buffer so we can seek backwards.
    targz_buffer = io.BytesIO(targz_request.raw.read())
    targz_file = tarfile.open(fileobj=targz_buffer, mode="r:*")

    ninja_log_file = targz_file.extractfile(f"logs/build/{project}/.ninja_log")
    ninja_log_text = ninja_log_file.raw.read().decode("utf8")

    return ninja_log_text


def get_child_jobs(baseurl):
    """Get a list of child jobs for a parent job."""
    ChildJob = collections.namedtuple("ChildJob", ["name", "job_nr"])
    regex = r"Starting building: .* » (.*) #(\d+)\n"
    console_text = get_console_text(baseurl)
    matches = [ChildJob(*m) for m in re.findall(regex, console_text)]
    matches = sorted(matches)
    return matches


def get_test_time(baseurl, project):
    """Get the test time for a specific project for a child job."""
    regex = re.compile(
        f"Test project (?:(?:/[^/\n]*)*)/{project}(?:.|\n)*?Total Test time \(real\) =(?:\s+)(\d+\.\d+)"
    )
    console_text = get_console_text(baseurl)
    matches = [float(m) for m in re.findall(regex, console_text)]
    assert (len(matches) <= 1)
    if len(matches) == 1:
        return matches[0]
    return "N/A"


def get_build_time(baseurl, project):
    """Get the build time for a specific project for a child job."""
    regex = re.compile("(\d+)\t(\d+)\t(?:.*)+\n")
    ninja_log = get_ninja_log(baseurl, project)
    min_millis = min([int(m[0]) for m in re.findall(regex, ninja_log)])
    max_millis = max([int(m[1]) for m in re.findall(regex, ninja_log)])
    return (max_millis - min_millis) / 1000.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "baseurl",
        help="Jenkins job URL, can be either a parent job or a child job")
    parser.add_argument("--test-time",
                        action="store_true",
                        help="Print test times")
    parser.add_argument("--build-time",
                        action="store_true",
                        help="Print build times")
    parser.add_argument("--project",
                        type=str,
                        default="popart",
                        help="Get results for specific project")
    parser.add_argument("--minimal",
                        action="store_true",
                        help="Output just the numbers, tab seperated")
    args = parser.parse_args()
    baseurl = args.baseurl

    cjs = get_child_jobs(baseurl)
    jobs = [
        f"https://jenkins.sourcevertex.net/job/poplar/job/{cj.name}/{cj.job_nr}"
        for cj in cjs
    ]

    if len(jobs) == 0:
        jobs = [baseurl]

    no_switches = (not args.test_time and not args.build_time)

    if args.test_time or no_switches:
        print(f"Test times for '{args.project}':")
        if not args.minimal:
            for job in jobs:
                test_time = get_test_time(job, args.project)
                print(f"- {job} test_time: {test_time}")
        else:
            test_times = "\t".join(
                [str(get_test_time(job, args.project)) for job in jobs])
            print(test_times)

    if args.build_time or no_switches:
        """ Print build times for every child job. """
        print(f"Build times for '{args.project}':")
        if not args.minimal:
            for job in jobs:
                build_time = get_build_time(job, args.project)
                print(f'- {job} build_time: {build_time}')
        else:
            build_times = "\t".join(
                [str(get_build_time(job, args.project)) for job in jobs])
            print(build_times)


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
