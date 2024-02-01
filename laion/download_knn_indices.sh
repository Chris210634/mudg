#!/bin/bash
for n in {00..54}; 
do
    wget https://the-eye.eu/public/AI/cah/laion5b/indices/vit-l-14/laion2B-en-imagePQ128/knn.index$n
done