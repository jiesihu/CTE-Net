for runs_file in Jun26_10-06-31_hitlab-SYS-7048GR-TR
do
    echo $runs_file
    Model_=CTE_Net
    python evaluation.py --model_path ./$Model_/runs/$runs_file --GPU_id 2 --dataset val
    python evaluation.py --model_path ./$Model_/runs/$runs_file --GPU_id 2 --dataset test
    python Compute_Metric.py --output_path ./$Model_/runs/$runs_file/output
    python Compute_Metric_test.py --output_path ./$Model_/runs/$runs_file/output
done