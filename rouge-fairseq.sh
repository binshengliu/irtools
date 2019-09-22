#!/bin/bash

if ! command -v rouge > /dev/null; then
    echo "Please install rouge" 2>&1
    echo "pip install --user rouge" 2>&1
    exit 1
fi

file=$1
if [ ! -f "$file" ]; then
    echo "Please specify fairseq-generate/fairseq-interactive output file" 2>&1
    exit 1
fi

hyp=$(mktemp --suffix .hyp)
ref=$(mktemp --suffix .ref)
grep '^H' "$file" | cut -f3- > $hyp
grep '^T' "$file" | cut -f2- > $ref
rouge -f $hyp $ref --avg
rm -f $hyp $ref
