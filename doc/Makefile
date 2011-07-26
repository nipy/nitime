# Makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
PAPER         =

# Internal variables.
PAPEROPT_a4     = -D latex_paper_size=a4
PAPEROPT_letter = -D latex_paper_size=letter
ALLSPHINXOPTS   = -d _build/doctrees $(PAPEROPT_$(PAPER)) $(SPHINXOPTS) .

.PHONY: help clean html api htmlonly latex changes linkcheck doctest

help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  html      make the HTML documentation"
	@echo "  api       	make API documents only"
	@echo "  latex     make the LaTeX, you can set PAPER=a4 or PAPER=letter"
	@echo "  pdf		make <latex> and run the PDF generation"
	@echo "  changes   make an overview of all changed/added/deprecated" \
	      		"items (ChangeLog)"
	@echo "  linkcheck check all external links for integrity"
	@echo "  doctest   run all doctests embedded in the documentation"
	@echo "  sf_fer_perez copy html files to sourceforge (fer_perez only)"
	@echo "  sf_arokem copy html files to sourceforge (arokem only)"

clean:
	-rm -rf _build/* *~ api/generated
	-rm -rf build/* 
	-rm examples/fig/*
	-rm examples/*.rst

htmlonly:
	$(SPHINXBUILD) -b html $(ALLSPHINXOPTS) _build/html
	@echo
	@echo "Build finished. The HTML pages are in _build/html."

api:    
	python ../tools/build_modref_templates.py
	@echo "Build API docs finished."

html: rstexamples api htmlonly 
	@echo "Build HTML and API finished."

html-no-exec: rstexamples-no-exec api htmlonly 
	@echo "Build HTML and API finished."

latex: api
	$(SPHINXBUILD) -b latex $(ALLSPHINXOPTS) _build/latex
	@echo
	@echo "Build finished; the LaTeX files are in _build/latex."
	@echo "Run \`make all-pdf' or \`make all-ps' in that directory to" \
	      "run these through (pdf)latex."
pdf: latex
	cd _build/latex && make all-pdf

all: html pdf

changes:
	$(SPHINXBUILD) -b changes $(ALLSPHINXOPTS) _build/changes
	@echo
	@echo "The overview file is in _build/changes."

linkcheck:
	$(SPHINXBUILD) -b linkcheck $(ALLSPHINXOPTS) _build/linkcheck
	@echo
	@echo "Link check complete; look for any errors in the above output " \
	      "or in _build/linkcheck/output.txt."

doctest:
	$(SPHINXBUILD) -b doctest $(ALLSPHINXOPTS) _build/doctest
	@echo "Testing of doctests in the sources finished, look at the " \
	      "results in _build/doctest/output.txt."


rstexamples:
	../tools/make_examples.py

rstexamples-no-exec:
	../tools/make_examples.py --no-exec

# Sourceforge doesn't appear to have a way of copying the files
# without specifying a username.  So we'll probably have one target
# for each project admin
#
# Note: If several developers alternate commits, we may get permission errors
# from rsync with the upload targets. If that happens, use the *_full targets
# below which first wipe out the whole dir on SF and then re-upload.

sf_fer_perez:
	@echo "Copying html files to sourceforge..."
	rsync -avH --delete -e ssh _build/html/ fer_perez,nipy@web.sourceforge.net:htdocs/nitime

sf_arokem:
	@echo "Copying html files to sourceforge..."
	rsync -avH --delete -e ssh _build/html/ arokem,nipy@web.sourceforge.net:htdocs/nitime

# Targets that force a clean and re-upload
sf_fer_perez_clean:
	@echo "Cleaning up sourceforge site"
	ssh fer_perez,nipy@shell.sf.net "rm -rf /home/groups/n/ni/nipy/htdocs/nitime/*"

sf_fer_perez_full: sf_fer_perez_clean sf_fer_perez

sf_arokem_clean:
	@echo "Cleaning up sourceforge site"
	ssh arokem,nipy@shell.sf.net "rm -rf /home/groups/n/ni/nipy/htdocs/nitime/*"

sf_arokem_full: sf_arokem_clean sf_arokem
