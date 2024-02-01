#!/bin/bash

# retrieve from the index using each of the 8 augmented copies of the text prototype
echo N_WAFFLE=${N_WAFFLE}
echo N_NEAREST_NEIGHBORS=${N_NEAREST_NEIGHBORS}
echo N_NUMBER_CLUSTERS=${N_NUMBER_CLUSTERS}
echo TMP_DIR=${TMP_DIR}
echo NUMBER_OF_PROCESSES=${NUMBER_OF_PROCESSES}

for ((i=0; i<${NUMBER_OF_PROCESSES}; i++));
do
    # To save time each retrieval is done in a separate process
    echo python scripts/retrieve.py  --dataset $1 --proc_id ${i} --num_nearest_neighbors ${N_NEAREST_NEIGHBORS} --tmp_directory ${TMP_DIR} --n_waffle ${N_WAFFLE} --nproc ${NUMBER_OF_PROCESSES}
    python scripts/retrieve.py  --dataset $1 --proc_id ${i} --num_nearest_neighbors ${N_NEAREST_NEIGHBORS} --tmp_directory ${TMP_DIR} --n_waffle ${N_WAFFLE} --nproc ${NUMBER_OF_PROCESSES}&
    pids[${i}]=$!
done

# wait for all pids
for pid in ${pids[*]}; do
    wait $pid
done

echo sleep 15
sleep 15

# combine the retrieval results
echo python scripts/retrieve_combine.py --dataset $1 --n_waffle ${N_WAFFLE} --nproc ${NUMBER_OF_PROCESSES} --num_nearest_neighbors ${N_NEAREST_NEIGHBORS} --tmp_directory ${TMP_DIR}
python scripts/retrieve_combine.py --dataset $1 --n_waffle ${N_WAFFLE} --nproc ${NUMBER_OF_PROCESSES} --num_nearest_neighbors ${N_NEAREST_NEIGHBORS} --tmp_directory ${TMP_DIR}

echo sleep 15
sleep 15