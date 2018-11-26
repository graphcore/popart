import fnmatch
import re
import poponnx


def get_poplar_cpu_device():

    return poponnx.DeviceManager().createCpuDevice()


def get_ipu_model(compileIPUCode=True, numIPUs=1, tilesPerIPU=1216):

    options = {
        "compileIPUCode": compileIPUCode,
        'numIPUs': numIPUs,
        "tilesPerIPU": tilesPerIPU
    }
    return poponnx.DeviceManager().createIpuModelDevice(options)


def get_compute_sets_from_report(report):

    lines = report.split('\n')
    cs = [x for x in lines if re.search(r'  Step #\d+:', x)]
    cs = [x.split(":")[1].strip() for x in cs]
    cs = [re.sub(r' \(\d+ executions?\)$', '', x) for x in cs]
    return cs


def check_whitelist_entries_in_compute_sets(cs_list, whitelist):

    result = True
    fail_list = []
    wl = [x + '*' for x in whitelist]
    for cs in cs_list:
        if len([x for x in wl if fnmatch.fnmatch(cs, x)]) == 0:
            fail_list += [cs]
            result = False
    if not result:
        print("Failed to match " + str(fail_list))
    return result


def check_compute_sets_in_whitelist_entries(cs_list, whitelist):

    result = True
    fail_list = []
    wl = [x + '*' for x in whitelist]
    for x in wl:
        if len([cs for cs in cs_list if fnmatch.fnmatch(cs, x)]) == 0:
            fail_list += [x]
            result = False
    if not result:
        print("Failed to match " + str(fail_list))
    return result


def check_all_compute_sets_and_list(cs_list, whitelist):

    return (check_whitelist_entries_in_compute_sets(cs_list, whitelist)
            and check_compute_sets_in_whitelist_entries(cs_list, whitelist))


def get_compute_set_regex_count(regex, cs_list):

    return len([cs for cs in cs_list if re.search(regex, cs)])
