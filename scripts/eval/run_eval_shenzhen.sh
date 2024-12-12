export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0,1,2,3

nohup torchrun --nproc_per_node=4 dinov2/eval/classification/linear_glori.py \
--batch-size 64 \
--val-epochs 100 \
--epochs 100 \
--image-size 512 \
--save-checkpoint-frequency 20 \
--eval-period-epochs 20 \
--val-metric-type binary_auc \
--config-file /fast/yangz16/outputs/cxr-million/vit_large_outputs/ibot333_highres512/config.yaml \
--pretrained-weights /fast/yangz16/outputs/cxr-million/vit_large_outputs/ibot333_highres512/eval/training_249999/teacher_checkpoint.pth \
--output-dir /fast/yangz16/outputs/chexfound/shenzhen/ibot333_512_eval_shenzhen \
--train-dataset Shenzhen:split=TRAIN:root=/bulk/yangz16/eval/shenzhen \
--val-dataset Shenzhen:split=VAL:root=/bulk/yangz16/eval/shenzhen \
--test-dataset Shenzhen:split=TEST:root=/bulk/yangz16/eval/shenzhen \
 &> /fast/yangz16/outputs/chexfound/shenzhen/ibot333_512_eval_shenzhen.log &