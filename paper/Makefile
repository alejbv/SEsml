# GenSIE 2026 SEsml paper - Makefile for CEURART compilation
# Requires: texlive (latex, pdflatex), ceurart.cls in same directory

TEX = gensie2026-sesml.tex
AUX = gensie2026-sesml.aux
BIB = references.bib
PDF = gensie2026-sesml.pdf

.PHONY: all clean distclean view

all: $(PDF)

$(PDF): $(TEX) $(BIB) figure-1-architecture.png ceurart.cls
	pdflatex $(TEX)
	pdflatex $(TEX)
	bibtex gensie2026-sesml
	pdflatex $(TEX)
	pdflatex $(TEX)

quick: $(TEX)
	pdflatex $(TEX)

clean:
	rm -f *.aux *.log *.out *.bbl *.blg *.run.xml *.bcf *.synctex.gz
	rm -f *.xmpdata *.xmpi *.abs

distclean: clean
	rm -f $(PDF)

view: $(PDF)
	open $(PDF) 2>/dev/null || xdg-open $(PDF) 2>/dev/null || echo "Open $(PDF) manually"
