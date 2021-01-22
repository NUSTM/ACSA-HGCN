export BERT_BASE_DIR=/home/hjcai/BERT/uncased_L-12_H-768_A-12
export DATA_DIR=/home/hjcai/BERT/pytorch_pretrained_BERT/ACSA-HGCN
export TASK_NAME=JCSC
export MODEL=GCN
export DOMAIN=laptop
export YEAR=2015

echo $BERT_BASE_DIR
python run_classifier_gcn.py \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --domain_type $DOMAIN \
  --year $YEAR \
  --model_type $MODEL\
  --do_lower_case \
  --data_dir $DATA_DIR \
  --bert_model $BERT_BASE_DIR \
  --max_seq_length 128 \
  --train_batch_size 8 \
  --learning_rate 5e-5 \
  --num_train_epochs 30.0 \
  --output_dir /home/hjcai/BERT/pytorch_pretrained_BERT/output/JointCategorySentiment/$DOMAIN$YEAR/JCSC/GCN