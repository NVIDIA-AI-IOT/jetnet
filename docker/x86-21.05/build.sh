#!/bin/bash

docker build -t jaybdub/jetnet:x86-21.05 -f $(pwd)/docker/x86-21.05/Dockerfile $(pwd)/docker/x86-21.05
