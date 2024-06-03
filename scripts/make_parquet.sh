#!/bin/bash

echo TMP_DIR=${TMP_DIR}
echo NUMBER_OF_PROCESSES=${NUMBzER_OF_PROCESSES}

for ((i=0; i<${NUMBER_OF_PROCESSES}; i++));
do
    echo python scripts/make_parquet_sub.py --dataset $1 --proc_id ${i} --tmp_directory ${TMP_DIR} --nproc ${NUMBER_OF_PROCESSES}
    python scripts/make_parquet_sub.py --dataset $1 --proc_id ${i} --tmp_directory ${TMP_DIR} --nproc ${NUMBER_OF_PROCESSES} &
    pids[${i}]=$!
done

# wait for all pids
for pid in ${pids[*]}; do
    wait $pid
done

echo sleep 15
sleep 15

echo python scripts/make_parquet_combine.py --dataset $1 --tmp_directory ${TMP_DIR} --nproc ${NUMBER_OF_PROCESSES}
python scripts/make_parquet_combine.py --dataset $1 --tmp_directory ${TMP_DIR} --nproc ${NUMBER_OF_PROCESSES}

echo sleep 15
sleep 15