#!/bin/bash

INSTALL_DIR="$(pwd)/install"

export MAX_JOBS=$(nproc)

bash ./build.sh \
    --install-dir ${INSTALL_DIR} \
    --enable-debug OFF 

pushd "${INSTALL_DIR}/bin"
./test_main
popd