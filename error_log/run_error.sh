cd ../
for dataset in "yahoomusic"
do
    mkdir error_log/${dataset} -p
    for method in "pq" "rq" "opq"
    do
        echo " writing into error_log/${dataset}/${method}.log"
        python run_error.py ${dataset} ${method} >> "error_log/${dataset}/${method}.log"
    done
done
