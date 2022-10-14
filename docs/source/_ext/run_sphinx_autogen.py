import sys
import os
from pathlib import Path
import subprocess


def run_autogen(_):
    cmd_path = "sphinx-autogen"
    if hasattr(sys, "real_prefix"):  # Check to see if we are in a virtualenv
        # If we are, assemble the path manually
        cmd_path = os.path.abspath(os.path.join(sys.prefix, "bin", cmd_path))
    # get absolute path
    current_dir = Path(__file__).resolve().parent
    docs_src = current_dir.parent
    templates = docs_src / "_templates"
    generated = docs_src / "api" / "generated"
    api_index = docs_src / "api" / "index.rst"
    subprocess.check_call(
        [cmd_path, "-i", "-t", str(templates), "-o", str(generated), str(api_index)]
    )

def setup(app):
    app.connect("builder-inited", run_autogen)
    return {
        'version': '0.1',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
