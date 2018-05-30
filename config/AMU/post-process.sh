#!/usr/bin/env bash

cat "${1:-/dev/stdin}" | sed "s/@@ //g" | scripts/moses/detruecase.perl | scripts/moses/deescape-special-chars.perl
