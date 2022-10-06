#!/bin/bash

emcc \
--bind \
-s ALLOW_MEMORY_GROWTH=1 \
-o jetnet_bindings.js \
src/js_bindings.cpp src/rle.cpp