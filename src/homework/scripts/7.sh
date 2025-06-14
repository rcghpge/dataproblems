#!/bin/bash
# Exercise 7: Clean and sum four numeric fields from World University Ranking

cut -d',' -f4,5,6,7 zips/world_university_rankings.csv | \
  tail -n +2 | \
  sed 's/[“”"'"'"']//g' | \
  grep -E '^[0-9]+(\.[0-9]+)?(,[0-9]+(\.[0-9]+)?){3}$' | \
  tr ',' '+' | \
  bc > hwzips/university_score_sums.txt
          
