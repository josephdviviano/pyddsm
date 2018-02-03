#!/bin/bash

echo "all data will be downloaded to: ${PWD}"

# downloading achieved via anon ftp (ncftpget), links tested Jan 2018
while read line; do
    echo "accessing: ${line} @ $(date)"
    ncftpget -R ${line}
done < ftp_list.txt

# clean up
find . -name *.gz -exec gunzip {} \;
