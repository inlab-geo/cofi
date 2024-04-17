"""Convert jupyter notebook to sphinx gallery notebook styled examples.
Dependencies:
pypandoc: install using `pip install pypandoc`

Adapted from source gist link below:
https://gist.github.com/chsasank/7218ca16f8d022e02a9c0deb94a310fe
"""

import os
import hashlib
from glob import glob
from shutil import copyfile, copytree, rmtree
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


def calculate_md5(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def check_md_file(md_file_path, original_file_path):
    original_file_md5 = calculate_md5(original_file_path)
    if not os.path.isfile(md_file_path):
        return False, original_file_md5
    with open(md_file_path, "r") as md_file:
        md_file_content = md_file.read().strip()
    return md_file_content == original_file_md5, original_file_md5

def convert_ipynb_to_gallery(full_file_name, dst_folder):
    # split full file path into example name with/without suffix
    file_name, example_name = file_name_without_path(full_file_name)
    
    # get the final script file path
    if dst_folder == TUTORIALS_SCRIPTS:
        script_file_folder = f"{dst_folder}"
    elif example_name in FIELD_DATA_EXAMPLES:
        script_file_folder = f"{dst_folder}_{FIELD_DATA}"
    else:
        script_file_folder = f"{dst_folder}_{SYNTH_DATA}"
    script_file_path = f"{script_file_folder}/{file_name}"
    script_file_path = script_file_path.replace(".ipynb", ".py")
    
    # calculate md5 hash and check whether file needs to be updated
    md_file = script_file_path.replace(".py", ".md5")
    cached, new_md5 = check_md_file(md_file, full_file_name)
    
    # convert ipynb to gallery script if needed
    if cached:
        print(f"File cached: {full_file_name}")
    else:
        print(f"Converting file: {full_file_name}")
        _convert_ipynb_to_gallery(full_file_name, script_file_path)
        with open(md_file, "w") as file:
            file.write(new_md5)

def _convert_ipynb_to_gallery(full_file_name, script_file_path):
    python_file = ""

    nb_dict = json.load(open(full_file_name))
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
        print(f"Date file from {data_file} to {dest_file_path}")
        copyfile(data_file, dest_file_path)
    # move illustrations folder
    if os.path.exists(f"{src_folder}/illustrations"):
        copy_and_overwrite(
            f"{src_folder}/illustrations", f"{dst_folder}/illustrations"
        )

def copy_and_overwrite(from_path, to_path):
    if os.path.exists(to_path):
        rmtree(to_path)
    copytree(from_path, to_path)
    print(f"Folder from {from_path} to {to_path}")

def gen_scripts_all(_):
    # #### TUTORIALS ####
    print("Generating tutorials gallery scripts...")
    # collect tutorials to convert to sphinx gallery scripts
    all_tutorials_scripts = glob(f"{TUTORIALS_SRC_DIR}/*/*.ipynb")
    # convert
    print("Converting tutorial files...")
    for script in all_tutorials_scripts:
        convert_ipynb_to_gallery(script, TUTORIALS_SCRIPTS)
    # collect all data and library files to move to scripts/
    move_data_files(f"{TUTORIALS_SRC_DIR}/*", TUTORIALS_SCRIPTS)
    # #### EXAMPLES ####
    print("Generating examples gallery scripts...")
    # collect examples to convert to sphinx gallery scripts
    all_examples_scripts = glob(f"{EXAMPLES_SRC_DIR}/*/*.ipynb")
    all_examples_scripts = [name for name in all_examples_scripts if "lab" not in name]
    # convert
    print("Converting example files...")
    for script in all_examples_scripts:
        convert_ipynb_to_gallery(script, EXAMPLES_SCRIPTS)
    # collect all data and library files to move to scripts/field_data
    move_data_files(f"{EXAMPLES_SRC_DIR}/*", f"{EXAMPLES_SCRIPTS}_{FIELD_DATA}")
    # #### DATA & THEORY ####
    print("\nCopying data and theory files...")
    copy_and_overwrite(f"{cofi_examples_dir}/data", f"{docs_src}/data")
    copy_and_overwrite(f"{cofi_examples_dir}/theory", f"{docs_src}/theory")
    print("\nOK.")

def setup(app):
    app.connect("builder-inited", gen_scripts_all)
    return {
        'version': '0.1',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }


if __name__ == '__main__':
    gen_scripts_all(None)
