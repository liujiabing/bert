export DATA_DIR=/data/duobi/bert-session-cls/data
export BERT_BASE_DIR=/data/duobi/bert-session-cls/chinese_L-12_H-768_A-12
export TRAINED_CLASSIFIER=/data/duobi/bert-session-cls/ckpt_all_wo_cs/model.ckpt-39304

#CUDA_VISIBLE_DEVICES=1,3 
python run_classifier.py --task_name=session --do_predict=true --data_dir=$DATA_DIR --vocab_file=$BERT_BASE_DIR/vocab.txt --bert_config_file=$BERT_BASE_DIR/bert_config.json --init_checkpoint=$TRAINED_CLASSIFIER --max_seq_length=200 --output_dir=/data/duobi/bert-session-cls/test_output
cat test_output/test_results.tsv | python postprocess.py > tmp
paste data/test.tsv tmp > tmpp
#cat tmpp | awk -F'\t' '{if($4==$6)print $6}' | wc -l
#cat tmpp | awk -F'\t' '{if($4!=$6)print $0}' > tmppp

cat tmpp | awk -F'\t' '{if($4=="0" && $6=="0")print $0}' > pred_0_label_0.txt
cat tmpp | awk -F'\t' '{if($4=="0" && $6!="0")print $0}' > pred_1_label_0.txt
cat tmpp | awk -F'\t' '{if($4!="0" && $6!="0")print $0}' > pred_1_label_1.txt
cat tmpp | awk -F'\t' '{if($4!="0" && $6=="0")print $0}' > pred_0_label_1.txt
wc -l pred_*
