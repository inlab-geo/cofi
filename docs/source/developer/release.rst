Make a release
^^^^^^^^^^^^^^

This page is for maintainers of the CoFI project who have access to the
source code repository. 

PyPI
----

The repository has been set up to use GitHub Actions to automatically
make a release. The workflow file is located in the ``.github/workflows`` directory 
of the repository:
https://github.com/inlab-geo/cofi/blob/main/.github/workflows/publish_pypi.yml

The trigger to the workflow is: a push to the ``main`` branch when the version
file ``src/cofi/version.py`` is updated. Once you have changed the version file
and pushed the changes to the `main` branch on GitHub, the workflow will
automatically build the library, run the tests and publish the library to
PyPI. 

**Remember to update the CHANGELOG.md file with the new version number and
release date before making a release.**

You can examine the workflow status by going to the "Actions" tab of the
repository on GitHub. Here's a quick link to the workflow:
https://github.com/inlab-geo/cofi/actions/workflows/publish_pypi.yml. Fix any issue and 
run the workflow manually if your release run fails for some reason.

Conda Forge
-----------

The conda-forge feedstock for CoFI is located at:
https://github.com/conda-forge/cofi-feedstock.

It automatically detects and makes a pull request to update the version of the
CoFI package when a new release is made on PyPI. The pull request is made in feedstock 
repository. All you (as a maintainer) need to do is to merge the pull request if
there isn't any issue with the new version.
