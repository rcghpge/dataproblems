#!/bin/bash
# Exercise 5: Replace values in Housing.csv and write to new file

cat zips/Housing.csv \
| sed 's/yes/1/g' \
| sed 's/no/0/g' \
| sed 's/unfurnished/0/g' \
| sed 's/semi-furnished/2/g' \
| sed 's/,furnished/,1/g' > hwzips/housing_numeric.csv
