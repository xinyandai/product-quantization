cd ../
for dataset in "netflix"
do
    mkdir error_log/${dataset} -p
    for method in "pq" "rq" "opq"
    do
        python run_error.py ${dataset} ${method} >> "error_log/${dataset}/${method}.log"
    done
done