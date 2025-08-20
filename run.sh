#!/bin/bash

export MAX_JOBS=$(nproc)

bash ./build.sh \
    --install-dir $PWD/install \
    --enable-debug OFF

pushd "$PWD/install/bin"
./test_main
popd