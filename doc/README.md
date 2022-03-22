# CoFI Documentation

> Note that our latest documentation is now hosted by [readthedocs](https://readthedocs.org/) and can be accessed [**here**](https://cofi.readthedocs.io/en/latest/). 
> 
> Instructions below are only for development and testing of this documentation.

## How to build this documentation?

The file [environment.yml](environment.yml) specifies packages required for developing this documentation. 

1. To create a new `conda` environment from the file:

    ```console
    conda env create -f environment.yml
    ```

2. To build / update documentation from this `doc` folder:

    ```console
    cd <path-to-cofi>/doc
    make html
    ```

    The above command cleans up previous build files (if exist), updates API reference list and builds webpage files.

3. To open the documentation built locally, use your browser to open the file: `<path-to-cofi>/doc/_build/html/index.html`.

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
