#!/bin/bash
# Exercise 3: Split Heart_Disease_Prediction.csv into Absence and Presence

head -n 1 zips/Heart_Disease_Prediction.csv > hwzips/heart_absence.csv
head -n 1 zips/Heart_Disease_Prediction.csv > hwzips/heart_presence.csv

grep "Absence" zips/Heart_Disease_Prediction.csv >> hwzips/heart_absence.csv
grep "Presence" zips/Heart_Disease_Prediction.csv >> hwzips/heart_presence.csv
