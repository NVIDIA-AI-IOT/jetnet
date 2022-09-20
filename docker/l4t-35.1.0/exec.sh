#!/bin/bash


docker run --network host --gpus all --rm --name=jetnet -v $(pwd):/jetnet jaybdub/jetnet:l4t-35.1.0 bash -c "cd /jetnet && $@"
