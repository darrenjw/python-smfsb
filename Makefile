# Makefile

VERSION=0.1.7

FORCE:
	make install
	make test

build:
	python3 -m build

install:
	make build
	python3 -m pip install ./dist/smfsb-$(VERSION).tar.gz

test:
	pytest tests/

publish:
	make build
	python3 -m twine upload dist/*$(VERSION)*


edit:
	emacs Makefile *.toml *.md src/smfsb/*.py tests/*.py &


# eof

