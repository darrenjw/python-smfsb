# Makefile

VERSION=1.1.4

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

format:
	black .

check:
	ruff check --select N
	ruff check

commit:
	make format
	ruff check --select N
	make test
	git commit -a
	git push
	git pull
	git log|less


edit:
	emacs Makefile *.toml *.md src/smfsb/*.py tests/*.py &

todo:
	grep TODO: src/smfsb/*py tests/*.py demos/*.py


# eof

