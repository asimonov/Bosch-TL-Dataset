from data_utils import load_tl_extracts
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tl_classifier_cnn import TLClassifierCNN

# load data
desired_dim = (32,32)
data_dirs = ['data/tl-extract-train', 'data/tl-extract-test', 'data/tl-extract-additional']
# OpenCV uses H, W, C. and C is ordered BGR
x, y = load_tl_extracts(data_dirs, desired_dim)

# transform data
relevant = ['green','off','yellow','red']
x2 = x[np.isin(y, relevant)]
y2 = y[np.isin(y, relevant)]

y2_set = set(y2)
y2_dict = {y: i for i,y in enumerate(y2_set)}
y2_n = [y2_dict[el] for el in y2]
n_classes = len(y2_set)

x = x2
y = y2

encoder = LabelBinarizer()
encoder.fit(y)
y_onehot = encoder.transform(y)
#y_onehot = y_onehot.astype(np.float32)

# split into train/test
pct_train = 85.
pct_valid = 15.
random_state = 123

train_features, val_features, train_labels, val_labels = train_test_split(
                                                            x, y_onehot,
                                                            train_size = pct_train/100.,
                                                            test_size = pct_valid/100.,
                                                            random_state = random_state)


tsc = TLClassifierCNN()

features_shape = ((None,) + train_features.shape[1:])
labels_shape = (None,train_labels.shape[1],)
#features_shape = (None,32,32,3)
#labels_shape = (None,4)

# define model
tsc.define_model(features_shape=features_shape, labels_shape=labels_shape)
model_param_file = 'ckpt/model.ckpt'
summary_dir = 'train_summaries'
tsc.set_save_files(model_param_file, summary_dir)

# learning parameters
epochs = 10
batch_size = 150
learning_rate = 0.001
max_iterations_without_improvement = 10
dropout_keep_probability=0.7

# create Tensorflow session
tsc.create_session(learning_rate)
# i have trained model several times, restarting from where I left off, sometimes changing parameters
tsc.restore_variables()

# main training
loss_epoch, train_acc_epoch, valid_acc_epoch, best_validation_accuracy = \
    tsc.train(train_images      = train_features,
              train_labels      = train_labels,
              validation_images = val_features,
              validation_labels = val_labels,
              dropout_keep_probability = dropout_keep_probability,
              batch_size        = batch_size,
              epochs            = epochs,
              max_iterations_without_improvement = max_iterations_without_improvement)

tsc.close_session()
