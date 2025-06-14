#!/bin/bash
# Exercise 4: Fraction of cars with no accidents

total=$(wc -l < zips/car_web_scraped_dataset.csv)
no_accidents=$(grep -i "no accident" zips/car_web_scraped_dataset.csv | wc -l)

echo "scale=4; $no_accidents / $total" | bc > hwzips/no_accidents_fraction.txt

