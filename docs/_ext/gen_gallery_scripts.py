"""Convert jupyter notebook to sphinx gallery notebook styled examples.
Usage: python ipynb_to_gallery.py <notebook.ipynb>
Dependencies:
pypandoc: install using `pip install pypandoc`

Adapted from source gist link below:
https://gist.github.com/chsasank/7218ca16f8d022e02a9c0deb94a310fe
"""

import sys
from glob import glob
from shutil import copyfile
from pathlib import Path

import pypandoc as pdoc
import json


current_dir = Path(__file__).resolve().parent
docs_dir = current_dir.parent
cofi_examples_dir = docs_dir / "cofi-examples"
NOTEBOOKS = "notebooks"
NOTEBOOKS_DIR = str(cofi_examples_dir / NOTEBOOKS)
EXAMPLES = "examples"
SCRIPTS_DIR = str(docs_dir / EXAMPLES / "scripts")

BADGE_BEGIN = "<!--<badge>-->"
BADGE_END = "<!--</badge>-->"

def convert_ipynb_to_gallery(file_name):
    python_file = ""

    nb_dict = json.load(open(file_name))
    cells = nb_dict['cells']

    for i, cell in enumerate(cells):
        if i == 0:  
            assert cell['cell_type'] == 'markdown', \
                'First cell has to be markdown'

            md_source = ''.join(cell['source'])
            rst_source = pdoc.convert_text(md_source, 'rst', 'md')
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
                    rst_source = pdoc.convert_text(md_source, 'rst', 'md')
                commented_source = '\n'.join(['# ' + x for x in
                                              rst_source.split('\n')])
                python_file = python_file + '\n\n\n' + '#' * 70 + '\n' + \
                    commented_source
            elif cell['cell_type'] == 'code':
                source = ''.join(cell['source'])
                python_file = python_file + '\n' * 2 + source
                python_file = python_file + '\n' * 2 + '#' * 70 + '\n' + '#'

    python_file = python_file.replace("\n%", "\n# %")

    file_name_without_path = file_name.split("/")[-1]
    script_file_path = f"{SCRIPTS_DIR}/{file_name_without_path}"
    script_file_path = script_file_path.replace(".ipynb", ".py")
    open(script_file_path, 'w').write(python_file)

def gen_scripts_all(_):
    print("Generating gallery scripts...")
    # collect notebooks to convert to sphinx gallery scripts
    all_scripts = glob(f"{NOTEBOOKS_DIR}/*/*.ipynb")
    all_scripts = [name for name in all_scripts if "lab" not in name]
    # convert
    print("Converting files...")
    for script in all_scripts:
        print(f"file: {script}")
        convert_ipynb_to_gallery(script)
    # collect all data and library files to move to scripts/
    all_data = glob(f"{NOTEBOOKS_DIR}/*/*.npz")
    all_data.extend(glob(f"{NOTEBOOKS_DIR}/*/*.dat"))
    all_data.extend(glob(f"{NOTEBOOKS_DIR}/*/*.csv"))
    all_data.extend(glob(f"{NOTEBOOKS_DIR}/*/*_lib.py"))
    # move
    print("\nMoving data files...")
    for data_file in all_data:
        data_filename_without_path = data_file.split("/")[-1]
        dest_file_path = f"{SCRIPTS_DIR}/{data_filename_without_path}"
        copyfile(data_file, dest_file_path)
    print("\nOK.")

def setup(app):
    app.connect("builder-inited", gen_scripts_all)
    return {
        'version': '0.1',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }


if __name__ == '__main__':
    # collect notebooks to convert to sphinx gallery scripts
    if sys.argv[-1] == "all":
        all_scripts = glob(f"{NOTEBOOKS_DIR}/*/*.ipynb")
        all_scripts = [name for name in all_scripts if "lab" not in name]
    else:
        all_scripts = [sys.argv[-1]]
    # convert
    print("Converting files...")
    for script in all_scripts:
        print(f"file: {script}")
        convert_ipynb_to_gallery(script)
    # collect all data and library files to move to scripts/
    all_data = glob(f"{NOTEBOOKS_DIR}/*/*.npz")
    all_data.extend(glob(f"{NOTEBOOKS_DIR}/*/*.dat"))
    all_data.extend(glob(f"{NOTEBOOKS_DIR}/*/*.csv"))
    all_data.extend(glob(f"{NOTEBOOKS_DIR}/*/*_lib.py"))
    # move
    print("\nMoving data files...")
    for data_file in all_data:
        data_filename_without_path = data_file.split("/")[-1]
        dest_file_path = f"{SCRIPTS_DIR}/{data_filename_without_path}"
        copyfile(data_file, dest_file_path)
    print("\nOK.")
