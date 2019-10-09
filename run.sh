DIR=/block1/duobi
export DATA_DIR=$DIR/bert-session-cls/data
export BERT_BASE_DIR=$DIR/bert-session-cls/chinese_L-12_H-768_A-12
export TRAINED_CLASSIFIER=$DIR/bert-session-cls/1/model.ckpt-26000

#rm -r $DIR/bert-session-cls/session_output/model.ckpt-*;
#rm -r $DIR/bert-session-cls/session_output/checkpoint;
rm -r $DIR/bert-session-cls/session_output/
CUDA_VISIBLE_DEVICES=0,1 python run_classifier.py --task_name=session --do_train=true --do_eval=true --data_dir=$DATA_DIR --vocab_file=$BERT_BASE_DIR/vocab.txt --bert_config_file=$BERT_BASE_DIR/bert_config.json --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt --max_seq_length=200 --train_batch_size=32 --learning_rate=2e-5 --num_train_epochs=2.0 --output_dir=$DIR/bert-session-cls/session_output
#CUDA_VISIBLE_DEVICES=1,3 python run_classifier.py --task_name=session --do_eval=true --data_dir=$DATA_DIR --vocab_file=$BERT_BASE_DIR/vocab.txt --bert_config_file=$BERT_BASE_DIR/bert_config.json --init_checkpoint=$TRAINED_CLASSIFIER/model.ckpt-3 --max_seq_length=200 --output_dir=$DIR/bert-session-cls/test_output --train_batch_size=32
#CUDA_VISIBLE_DEVICES=0,1 python run_classifier.py --task_name=session --do_eval=true --do_predict=true --data_dir=$DATA_DIR --vocab_file=$BERT_BASE_DIR/vocab.txt --bert_config_file=$BERT_BASE_DIR/bert_config.json --init_checkpoint=$TRAINED_CLASSIFIER --max_seq_length=150 --output_dir=$DIR/bert-session-cls/test_output
