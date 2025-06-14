#!/bin/bash
# Exercise 2: Split diabetes dataset into 3 parts

head -n 1 zips/diabetes_prediction_dataset.csv > hwzips/diabetes_part1.csv
head -n 1 zips/diabetes_prediction_dataset.csv > hwzips/diabetes_part2.csv
head -n 1 zips/diabetes_prediction_dataset.csv > hwzips/diabetes_part3.csv

total=$(wc -l < zips/diabetes_prediction_dataset.csv)
data_lines=$((total - 1))
split_size=$((data_lines / 3))

tail -n +2 zips/diabetes_prediction_dataset.csv | head -n $split_size >> hwzips/diabetes_part1.csv
tail -n +$((split_size + 2)) zips/diabetes_prediction_dataset.csv | head -n $split_size >> hwzips/diabetes_part2.csv
tail -n +$((2 * split_size + 2)) zips/diabetes_prediction_dataset.csv >> hwzips/diabetes_part3.csv
