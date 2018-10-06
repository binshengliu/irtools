#!/bin/bash
each_server.sh 'for pid in $(ps aux | grep "^s3676608.*forkserver" | grep -v grep | cut -d" " -f 2) ; do prlimit -p $pid --nofile=4096: ; done'

