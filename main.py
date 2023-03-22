from utils import *
from model import *

# load all x_train and y_train data
x_train,y_train = load_pkl_data(train_pkl)
print("%d Train Data loaded"%(len(x_train)))
# load all validation data
x_val,y_val = load_pkl_data(validation_pkl)
print("%d Validation Data loaded"%(len(x_val)))

# load label maps
label2ind = load_dict_from_json(label2ind_json)
ind2label = load_dict_from_json(ind2label_json)
print("Labels loaded")

nc = len(label2ind.keys())
print("Total %d classes"%nc)

# make a model
model = AttentionClassifier(nc)
print("Model created")
model.load_model("model/")
model.fit(x_train[:200],y_train[:200],label2ind,validation_data=[x_val[:16],y_val[:16]],batch_size=32)
