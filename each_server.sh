#!/bin/sh
for id in 01 02 03 04 05 07 08 09 10; do
  echo segsresap$id 1>&2
  ssh segsresap$id $*
done
