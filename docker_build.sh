#!/bin/bash

# for preprocess
docker build -t vaetracer/preprocess:0.0.1 -f ./docker/preprocess/Dockerfile .
docker save vaetracer/preprocess:0.0.1 | gzip > vaetracer-preprocess.tar.gz
# # gunzip -c vaetracer-preprocess.tar.gz | docker load

# for scMut
docker build -t vaetracer/scmut:0.1.0 -f ./docker/scMut/Dockerfile .
docker save vaetracer/scmut:0.1.0 | gzip > vaetracer-scmut.tar.gz
# gunzip -c vaetracer-scmut.tar.gz | docker load


# for MutTracer
docker build -t vaetracer/muttracer:0.1.0 -f ./docker/MutTracer/Dockerfile .
docker save vaetracer/muttracer:0.1.0 | gzip > vaetracer-muttracer.tar.gz
# # gunzip -c vaetracer-muttracer.tar.gz | docker load