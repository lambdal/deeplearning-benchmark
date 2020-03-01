#!/usr/bin/env bash

set -e

DATADIR=${1:-"/data/tacotron2/LJSpeech-1.1"}
BZ2ARCHIVE="LJSpeech-1.1.tar.bz2"
ENDPOINT="http://data.keithito.com/data/speech/$BZ2ARCHIVE"


cd $DATADIR

if [ ! -f "${DATADIR}/${BZ2ARCHIVE}" ]; then
    echo "dataset archive is missing, downloading ..."
    wget "$ENDPOINT"
fi

tar jxvf "${DATADIR}/${BZ2ARCHIVE}"
