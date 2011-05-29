# Make file for generation of a talk about time-series analysis of fMRI data,
# using nitime

# Paths to the programs used by this Makefile.
PDFLATEX = pdflatex
BIBTEX = bibtex
MAKEINDEX = makeindex
CP = cp
RM = rm
GREP = grep
DIFF = diff -q
TOUCH = touch

# Paths to the input and output files.  These can be changed.
TEXFILE = nitime_talk.tex
PDFFILE = nitime_talk.pdf
DVIFILE = nitime_talk.dvi

# Paths to intermediate files that are created by the make process.
# Do not change these.
TEXLOG = $(basename $(TEXFILE)).log
BIBLIOGRAPHYFILE = $(basename $(TEXFILE)).bbl
BIBTEXLOG = $(basename $(TEXFILE)).blg
INDEXFILE = $(basename $(TEXFILE)).idx
MAKEINDEXLOG = $(basename $(TEXFILE)).ilg
NAVFILE = $(basename $(TEXFILE)).nav
SNMFILE = $(basename $(TEXFILE)).snm
AUXFILE = $(basename $(TEXFILE)).aux
DVIFILE = $(basename $(TEXFILE)).dvi
VRBFILE = $(basename $(TEXFILE)).vrb
OLDAUXFILE = ".$(AUXFILE).old"
TOCFILE = $(basename $(TEXFILE)).toc
OUTFILE = $(basename $(TEXFILE)).out
OLDINDEXITEMFILE = ".$(INDEXFILE).old"

# The primary target (created by the command "make") is the PDF file.
pdf:
	pdflatex $(TEXFILE)

all: clean pdf

clean:
	$(RM) -f $(AUXFILE) \
                 $(SNMFILE) \
                 $(NAVFILE) \
		 $(VRBFILE) \
                 $(DVIFILE) \
                 $(TEXLOG) \
                 $(TOCFILE) \
                 $(OUTFILE) \
                 $(BIBLIOGRAPHYFILE) \
                 $(BIBTEXLOG) \
                 $(INDEXFILE) \
                 $(INDEXITEMFILE) \
                 $(OLDINDEXITEMFILE) \
                 $(MAKEINDEXLOG) \
                 $(PDFFILE)
