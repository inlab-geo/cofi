# CoFI Documentation

> Note that our latest documentation is now hosted by [readthedocs](https://readthedocs.org/) and can be accessed [**here**](https://cofi.readthedocs.io/en/latest/). 
> 
> Instructions below are only for development and testing of this documentation.

## How to build this documentation?

The file [environment.yml](environment.yml) specifies packages required for developing this documentation. 

1. Clone `cofi` and update submodule (`cofi-examples`):
   
   ```console
   git clone https://github.com/inlab-geo/cofi.git
   cd cofi
   git submodule update --init
   ```

2. To create a new `conda` environment from the file:

    ```console
    conda env create -f environment.yml
    ```

3. To build your changes:

    ```console
    cd docs
    make html
    ```

    The above command cleans up previous build files (if exist), updates API reference list and builds webpage files.

4. Open your browser and go to file://\<path-to-cofi\>/docs/_build/html/index.html.
5. Redo step 2 after you've changed things in this "docs" folder.
6. To (only) update API references, use `make update_api`.
7. To (only) clean up built files, use `make clean`.

## Structure of this documentation

This documentation follows the guide [here](https://www.writethedocs.org/guide/writing/beginners-guide-to-docs/), so aims to include:

1. what `cofi` is & what problem it solves
2. small code example
3. a link to code & issue tracker
4. frequently asked questions
5. how to get support
6. information for people who want to contribute back
7. installation instructions
8. license
