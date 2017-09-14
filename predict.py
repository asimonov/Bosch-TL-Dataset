from data_utils import load_tl_extracts
#import numpy as np
from tl_classifier_cnn import TLClassifierCNN, TLLabelConverter

# load data
desired_dim = (32,32)
data_dirs = ['data/tl-extract-test']
x, y = load_tl_extracts(data_dirs, desired_dim)
# x is image in OpenCV imread format. pixels are uint8 from 0 to 255. shape is H, W, C. C is ordered BGR
# y here are strings like 'green' etc

# filter data with only labels relevant for us
converter = TLLabelConverter()
x, y = converter.filter(x, y)

# load the model
tlc = TLClassifierCNN()
model_dir = 'model'
tlc.load_model(model_dir)

import cv2
import numpy as np
image = cv2.imread('data/tl-extract-test/000001_green.png')
resized = cv2.resize(image, (32,32), interpolation=cv2.INTER_LINEAR)
assert (resized.shape == (32, 32, 3))
labels, probabilities = tlc.predict(np.array([resized]), batch_size=1)
if labels[0]=='green':
  print('correct')
else:
  print('incorrect')


# run predictions
batch_size = 50
labels, probs = tlc.predict(x, batch_size=batch_size)

# calculate accuracy
correct = sum([1 if y[i]==labels[i] else 0 for i in range(len(y))])
accuracy = float(correct) / len(y)
print('accuracy: {}. correct {} out of {}'.format(accuracy, correct, len(y)))



tlc.close_session()
