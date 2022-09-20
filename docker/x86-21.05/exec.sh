#!/bin/bash


docker run --network host --gpus all --rm --name=jetnet -v $(pwd):/jetnet jaybdub/jetnet:x86-21.05 bash -c "cd /jetnet && $@"
