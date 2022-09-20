#!/bin/bash


docker run --network host --gpus all --rm --name=jetnet -v $(pwd):/jetnet jaybdub/jetnet:l4t-34.1.1 bash -c "cd /jetnet && $@"
