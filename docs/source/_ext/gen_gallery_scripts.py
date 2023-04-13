"""Convert jupyter notebook to sphinx gallery notebook styled examples.
Dependencies:
pypandoc: install using `pip install pypandoc`

Adapted from source gist link below:
https://gist.github.com/chsasank/7218ca16f8d022e02a9c0deb94a310fe
"""

import sys
from glob import glob
from shutil import copyfile
from pathlib import Path
import json
import pypandoc


current_dir = Path(__file__).resolve().parent
docs_src = current_dir.parent
cofi_examples_dir = docs_src / "cofi-examples"
EXAMPLES = "examples"
EXAMPLES_SRC_DIR = str(cofi_examples_dir / EXAMPLES)
EXAMPLES_SCRIPTS = str(docs_src / EXAMPLES / "scripts")
TUTORIALS = "tutorials"
TUTORIALS_SRC_DIR = str(cofi_examples_dir / TUTORIALS)
TUTORIALS_SCRIPTS = str(docs_src / TUTORIALS / "scripts")
FIELD_DATA = "field_data"
SYNTH_DATA = "synth_data"

BADGE_BEGIN = "<!--<badge>-->"
BADGE_END = "<!--</badge>-->"

FIELD_DATA_EXAMPLES = []
with open(current_dir / "real_data_examples.txt", "r") as f:
    FIELD_DATA_EXAMPLES = f.read().splitlines() 

def convert_ipynb_to_gallery(file_name, dst_folder):
    python_file = ""

    nb_dict = json.load(open(file_name))
    cells = nb_dict['cells']

    for i, cell in enumerate(cells):
        if i == 0:  
            assert cell['cell_type'] == 'markdown', \
                'First cell has to be markdown'

            md_source = ''.join(cell['source'])
            rst_source = pypandoc.convert_text(md_source, 'rst', 'md')
            python_file = '"""\n' + rst_source + '\n"""'
        else:
            if cell['cell_type'] == 'markdown':
                md_source = ''.join(cell['source'])
                if md_source.startswith(BADGE_BEGIN) and md_source.endswith(BADGE_END):
                    rst_source = ".. raw:: html\n\n\t"
                    rst_source += md_source.replace(BADGE_BEGIN,"<badge>").replace(BADGE_END,"</badge>")
                elif "Table of contents" in md_source:
                    continue
                else:
                    rst_source = pypandoc.convert_text(md_source, 'rst', 'md')
                commented_source = '\n'.join(['# ' + x for x in
                                              rst_source.split('\n')])
                python_file = python_file + '\n\n\n' + '#' * 70 + '\n' + \
                    commented_source
            elif cell['cell_type'] == 'code':
                source = ''.join(cell['source'])
                python_file = python_file + '\n' * 2 + source
                python_file = python_file + '\n' * 2 + '#' * 70 + '\n' + '#'

    python_file = python_file.replace("\n%", "\n# %")
    python_file = python_file.replace("\n!", "\n# !")
    python_file += "\n# sphinx_gallery_thumbnail_number = -1"

    _file_name, _example_name = file_name_without_path(file_name)
    if dst_folder == TUTORIALS_SCRIPTS:
        script_file_path = f"{dst_folder}/{_file_name}"
    elif _example_name in FIELD_DATA_EXAMPLES:
        script_file_path = f"{dst_folder}/{FIELD_DATA}/{_file_name}"
    else:
        script_file_path = f"{dst_folder}/{SYNTH_DATA}/{_file_name}"
    script_file_path = script_file_path.replace(".ipynb", ".py")
    open(script_file_path, 'w').write(python_file)

def file_name_without_path(file_path):
    name = file_path.split("/")[-1]
    name_without_suffix = name.split(".")[0]
    return name, name_without_suffix

def move_data_files(src_folder, dst_folder):
    # collect all data and library files to move to dst_folder
    all_patterns = [
        "*.npz",
        "*.dat",
        "*.csv",
        "*.vtk",
        "*.txt",
        "*_lib.py",
    ]
    all_data = []
    for pattern in all_patterns:
        all_data.extend(glob(f"{src_folder}/{pattern}"))
    # move
    print("\nMoving data files...")
    for data_file in all_data:
        data_filename_without_path = data_file.split("/")[-1]
        dest_file_path = f"{dst_folder}/{data_filename_without_path}"
        copyfile(data_file, dest_file_path)

def gen_scripts_all(_):
    # #### TUTORIALS ####
    print("Generating tutorials gallery scripts...")
    # collect tutorials to convert to sphinx gallery scripts
    all_tutorials_scripts = glob(f"{TUTORIALS_SRC_DIR}/*.ipynb")
    # convert
    print("Converting tutorial files...")
    for script in all_tutorials_scripts:
        print(f"file: {script}")
        convert_ipynb_to_gallery(script, TUTORIALS_SCRIPTS)
    # collect all data and library files to move to scripts/
    move_data_files(TUTORIALS_SRC_DIR, TUTORIALS_SCRIPTS)
    # #### EXAMPLES ####
    print("Generating examples gallery scripts...")
    # collect examples to convert to sphinx gallery scripts
    all_examples_scripts = glob(f"{EXAMPLES_SRC_DIR}/*/*.ipynb")
    all_examples_scripts = [name for name in all_examples_scripts if "lab" not in name]
    # convert
    print("Converting example files...")
    for script in all_examples_scripts:
        print(f"file: {script}")
        convert_ipynb_to_gallery(script, EXAMPLES_SCRIPTS)
    # collect all data and library files to move to scripts/field_data
    move_data_files(f"{EXAMPLES_SRC_DIR}/*", f"{EXAMPLES_SCRIPTS}/{FIELD_DATA}")
    print("\nOK.")

def setup(app):
    app.connect("builder-inited", gen_scripts_all)
    return {
        'version': '0.1',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }


if __name__ == '__main__':
    # collect examples to convert to sphinx gallery scripts
    if sys.argv[-1] == "all":
        all_scripts = glob(f"{EXAMPLES_SRC_DIR}/*/*.ipynb")
        all_scripts = [name for name in all_scripts if "lab" not in name]
    else:
        all_scripts = [sys.argv[-1]]
    # convert
    print("Converting files...")
    for script in all_scripts:
        print(f"file: {script}")
        convert_ipynb_to_gallery(script, EXAMPLES_SCRIPTS)
    # collect all data and library files to move to scripts/
    all_data = glob(f"{EXAMPLES_SRC_DIR}/*/*.npz")
    all_data.extend(glob(f"{EXAMPLES_SRC_DIR}/*/*.dat"))
    all_data.extend(glob(f"{EXAMPLES_SRC_DIR}/*/*.csv"))
    all_data.extend(glob(f"{EXAMPLES_SRC_DIR}/*/*_lib.py"))
    # move
    print("\nMoving data files...")
    for data_file in all_data:
        data_filename_without_path = data_file.split("/")[-1]
        dest_file_path = f"{EXAMPLES_SCRIPTS}/{data_filename_without_path}"
        copyfile(data_file, dest_file_path)
    print("\nOK.")
