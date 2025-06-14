#!/bin/bash
# Exercise 8: Sort cancer patient data by age

head -n 1 zips/cancer_patient_data_sets.csv > hwzips/cancer_sorted.csv
tail -n +2 zips/cancer_patient_data_sets.csv | sort -t',' -k3,3n >> hwzips/cancer_sorted.csv

