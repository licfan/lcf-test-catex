#/usr/bin/env bash

# build sphinx document under doc/
mkdir -p doc/_sphinx_gen

# TODO: generate tools doc

# TODO: generate doc with bin file of catex

# TODO: generate package doc

# TODO: edit doc/index.rst to create catex website
# build html
sphinx-build -b html doc doc/build.html

touch doc/build/.nojekyll