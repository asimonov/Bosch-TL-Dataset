from data_utils import load_tl_extracts
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from tl_classifier_cnn import TLClassifierCNN, TLLabelConverter


# load data
desired_dim = (32,32)
data_dirs = ['data/tl-extract-train', 'data/tl-extract-test', 'data/tl-extract-additional']
x, y = load_tl_extracts(data_dirs, desired_dim)
# x is image in OpenCV imread format. pixels are uint8 from 0 to 255. shape is H, W, C. C is ordered BGR
# y here are strings like 'green' etc

# filter data with only labels relevant for us
converter = TLLabelConverter()
x, y = converter.filter(x, y)



features_shape = (None,32,32,3)
labels_shape = (None, 4)
save_file = 'ckpt/model.ckpt'

tlc = TLClassifierCNN(features_shape, labels_shape, save_file)
tlc.restore_checkpoint()

batch_size = 50

labels, probs = tlc.predict(x, batch_size=batch_size)

tlc.close_session()
