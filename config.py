# https://tfhub.dev/google/collections/bert/1 load models from here
hub_preprocessor_model = "https://tfhub.dev/tensorflow/bert_en_cased_preprocess/3"
hub_transformer_model = "https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/4"
hf_preprocessor = "prajjwal1/bert-tiny"
hf_encoder = "prajjwal1/bert-tiny"
data_root = "data"
# for colab
# data_root = "/content/drive/MyDrive/Test_Executions/hate_speech_classification/data"
train_csv_file = "%s/hate_speech_train.csv" % data_root
label2ind_json = "%s/label2ind.json" % data_root
ind2label_json = "%s/ind2label.json" % data_root
attention_dim = 128  # should come from transformer dim, H value
train_pkl = "%s/train.pkl" % data_root
test_pkl = "%s/test.pkl" % data_root
validation_pkl = "%s/validation.pkl" % data_root
learning_rate = 1e-5
