# CoFI Documentation

> Note that our latest documentation is now hosted by [readthedocs](https://readthedocs.org/) and can be accessed [**here**](https://cofi.readthedocs.io/en/latest/). 
> 
> Instructions below are only for development and testing purposes.

## How to build this documentation?

`sphinx` and `sphinx-insipid-theme` are required for building this documentation.

So if you are in a hurry, simply use `pip install sphinx sphinx-insipid-theme` to ensure the dependencies.

Alternatively, the *recommended* way is to use a virtual environment so that it doesn't conflict with the dependencies of your other Python programs. We have listed all the dependencies required for developing CoFI in a file (`environment_dev.yml`) at the root level of this repository. 

### Building for the first time

```bash
# clone the repo
cd <path-where-you-want-cofi-to-be-in>
git clone https://github.com/inlab-geo/cofi.git

# create a virtual environment
cd cofi
conda env create -f environment_dev.yml
conda activate cofi_dev

# build the documentation
cd doc
make html
```

### Updating the documentation
If you already have cofi repository cloned and virtual environment ready, then do the following:

```bash
cd <path-where-you-have-your-cofi>/cofi
conda activate cofi_dev
make html
```

Then use your browser to open the `index.html` file located in: `<path-of-cofi>/doc/_build/html/`.
