#!/bin/bash

echo "Check that the granatum path is set correctly:"
echo $GRANATUM_SWD
echo ""
echo "I am currently in directory..."
res=`pwd`
echo "$res"
echo ""
echo "Running color scatterplot"
python ./expressiondensityplot.py
