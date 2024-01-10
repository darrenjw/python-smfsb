# Development info

```bash
# build:
python3 -m build
# local install:
python3 -m pip install ./dist/smfsb-0.0.4.tar.gz
# PyPI upload:
python3 -m twine upload dist/*0.0.4*

# Upgrade installed package:
pip install --upgrade package-name
```

Most of this logic is in the `Makefile`.

After publishing on PyPI, also do a GitHub release.

After the GitHub release, bump version number in the `Makefile` and `pyproject.toml`

Python packaging tutorials: https://packaging.python.org/en/latest/tutorials/



## Documentation

From the `docs` folder:
```bash
sphinx-apidoc -o source ../src/smfsb/
# to create stubs. Then
make clean
make html

xdg-open _build/html/index.html
```

