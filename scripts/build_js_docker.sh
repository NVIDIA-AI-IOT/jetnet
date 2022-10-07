#!/bin/bash

docker run --rm -v $(pwd):/src trzeci/emscripten ./scripts/build_js.sh