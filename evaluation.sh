for training_dir in Jun26_10-06-31_hitlab-SYS-7048GR-TR
do
    echo $training_dir
    Model_=CTE_Net
    python evaluation.py --model_path ./$Model_/runs/$training_dir --GPU_id 2 --dataset val
    python evaluation.py --model_path ./$Model_/runs/$training_dir --GPU_id 2 --dataset test
    python Compute_Metric.py --output_path ./$Model_/runs/$training_dir/output
    python Compute_Metric_test.py --output_path ./$Model_/runs/$training_dir/output
done
