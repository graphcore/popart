import subprocess
import sys

import c10datadir
import download_dataset.common

MB_REQUIRED = 400


def enough_hardrive_space():
    df_out = subprocess.check_output(
        ["df", "-m", str(c10datadir.c10datadir)]).decode("utf-8")
    second_line = df_out.strip().split("\n")[-1]

    cols = [col for col in second_line.split(" ") if col.strip() != ""]
    mb_remaining = int(cols[3])

    return (mb_remaining >= MB_REQUIRED)


if not c10datadir.c10datadir.exists():
    print(f'Creating directory {c10datadir.c10datadir}', flush=True)
    c10datadir.c10datadir.mkdir()
    if not enough_hardrive_space():
        print(f'Not enough disk space on {c10datadir.c10datadir}')
        sys.exit(1)
else:
    print(f'Using existing directory {c10datadir.c10datadir}', flush=True)
    download_dataset.common.report_skipped()
