#!/bin/bash
# Unzips all .zip files in the current directory into the same directory

for f in *.zip; do
  echo "Unzipping $f..."
  unzip -o "$f" -d ./
done

echo " All ZIP files have been extracted."

