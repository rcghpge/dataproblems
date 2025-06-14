#!/bin/bash
# Exercise 6: Remove CustomerID column

cut -d',' -f2- zips/mall_customers.csv > hwzips/mall_customers_no_id.csv
