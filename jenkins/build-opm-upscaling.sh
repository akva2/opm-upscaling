#!/bin/bash

function build_opm_upscaling {
  # Build ERT
  pushd .
  mkdir -p $WORKSPACE/deps/ert
  cd $WORKSPACE/deps/ert
  git init .
  git remote add origin https://github.com/Ensembles/ert
  git fetch origin $ERT_REVISION:branch_to_build
  test $? -eq 0 || exit 1
  git checkout branch_to_build
  popd

  pushd .
  mkdir -p serial/build-ert
  cd serial/build-ert
  cmake $WORKSPACE/deps/ert/devel -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$WORKSPACE/serial/install
  cmake --build . --target install
  popd

  # Build opm-common
  pushd .
  mkdir -p $WORKSPACE/deps/opm-common
  cd $WORKSPACE/deps/opm-common
  git init .
  git remote add origin https://github.com/OPM/opm-common
  git fetch origin $OPM_COMMON_REVISION:branch_to_build
  test $? -eq 0 || exit 1
  git checkout branch_to_build
  popd
  source $WORKSPACE/deps/opm-common/jenkins/build-opm-module.sh

  pushd .
  mkdir serial/build-opm-common
  cd serial/build-opm-common
  build_module "-DCMAKE_INSTALL_PREFIX=$WORKSPACE/serial/install" 0 $WORKSPACE/deps/opm-common
  popd

  # Build opm-parser
  clone_and_build_module opm-parser "-DCMAKE_PREFIX_PATH=$WORKSPACE/serial/install -DCMAKE_INSTALL_PREFIX=$WORKSPACE/serial/install" $OPM_PARSER_REVISION $WORKSPACE/serial

  # Build opm-material
  clone_and_build_module opm-material "-DCMAKE_PREFIX_PATH=$WORKSPACE/serial/install -DCMAKE_INSTALL_PREFIX=$WORKSPACE/serial/install" $OPM_MATERIAL_REVISION $WORKSPACE/serial

  # Build opm-core
  clone_and_build_module opm-core "-DCMAKE_PREFIX_PATH=$WORKSPACE/serial/install -DCMAKE_INSTALL_PREFIX=$WORKSPACE/serial/install" $OPM_CORE_REVISION $WORKSPACE/serial

  # Build opm-grid
  clone_and_build_module opm-grid "-DCMAKE_PREFIX_PATH=$WORKSPACE/serial/install -DCMAKE_INSTALL_PREFIX=$WORKSPACE/serial/install" $OPM_GRID_REVISION $WORKSPACE/serial

  # Build opm-output
  clone_and_build_module opm-output "-DCMAKE_PREFIX_PATH=$WORKSPACE/serial/install -DCMAKE_INSTALL_PREFIX=$WORKSPACE/serial/install" $OPM_OUTPUT_REVISION $WORKSPACE/serial

  # Build opm-upscaling
  pushd .
  mkdir serial/build-opm-upscaling
  cd serial/build-opm-upscaling
  build_module "-DCMAKE_PREFIX_PATH=$WORKSPACE/serial/install" 1 $WORKSPACE
  popd
}
