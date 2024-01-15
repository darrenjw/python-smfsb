# Makefile

VERSION=0.1.2

FORCE:
	make build

build:
	python3 -m build

install:
	make build
	python3 -m pip install ./dist/smfsb-$(VERSION).tar.gz

publish:
	make build
	python3 -m twine upload dist/*$(VERSION)*


edit:
	emacs Makefile *.toml *.md src/smfsb/*.py &


# eof

