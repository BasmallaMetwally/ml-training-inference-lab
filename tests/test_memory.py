import json
import subprocess
import sys

import pytest

from ml_lab.memory import parse_proc_status


def test_proc_status_units_and_required_fields():
    assert parse_proc_status("Name:\tpython\nVmRSS:\t123 kB\nVmHWM:\t456 kB\n") == {
        "VmRSS": 123 * 1024,
        "VmHWM": 456 * 1024,
    }
    with pytest.raises(ValueError):
        parse_proc_status("VmRSS: 123 kB\n")


@pytest.mark.skipif(sys.platform != "linux", reason="Linux exec-local /proc regression")
def test_parent_allocation_does_not_contaminate_child_vm_hwm():
    child = (
        "from ml_lab.memory import process_memory; import json; print(json.dumps(process_memory()))"
    )
    parent = "import subprocess,sys; payload=bytearray(160*1024*1024); print(subprocess.check_output([sys.executable,'-c',sys.argv[1]],text=True))"
    output = subprocess.check_output([sys.executable, "-c", parent, child], text=True)
    result = json.loads(output)
    assert result["method"] == "/proc/self/status VmHWM"
    assert result["peak_rss_bytes"] < 100 * 1024 * 1024
