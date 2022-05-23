# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXAPIDOC  = sphinx-apidoc
SPHINXAUTOGEN = sphinx-autogen
SOURCEDIR     = .
BUILDDIR      = _build
APIDIR        = api
CODEDIR       = ../cofi/
NOTEBOOKDIR   = cofi-examples


all: html

clean:
	@echo
	@echo "Cleaning up..."
	@echo
	rm -rf $(BUILDDIR)/html/*
	rm -rf $(BUILDDIR)/doctrees
	rm -rf $(BUILDDIR)/doctest
	rm -rf $(APIDIR)/generated
	rm -rf $(NOTEBOOKDIR)/generated

update_api: clean
	@echo
	@echo "Building API docs..."
	@echo
	$(SPHINXAUTOGEN) -i -t _templates -o api/generated api/index.rst

html: clean
	@echo
	@echo "Building HTML files..."
	@echo
	$(SPHINXBUILD) -d $(BUILDDIR)/doctrees -b html . $(BUILDDIR)/html
	@echo

# pdf: clean
# 	@echo
# 	@echo "Building PDF files..."
# 	@echo
# 	$(SPHINXBUILD) -d $(BUILDDIR)/doctrees -b latexpdf . $(BUILDDIR)/pdf
# 	@echo

linkcheck: update_api
	@echo
	@echo "Checking links..."
	@echo
	$(SPHINXBUILD) -b linkcheck . $(BUILDDIR)/linkcheck

doctest: 
	@echo
	@echo "Performing doc tests..."
	@echo
	$(SPHINXBUILD) -b doctest . $(BUILDER)/doctest


# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
