#!/bin/bash


docker build -t jaybdub/jetnet:l4t-35.1.0 -f $(pwd)/docker/l4t-35.1.0/Dockerfile $(pwd)/docker/l4t-35.1.0
