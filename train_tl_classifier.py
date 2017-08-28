from tl_classifier_cnn import TLClassifierCNN

tsc = TLClassifierCNN()

features_shape = (None,32,32,3)
labels_shape = (None,4)

# define model
tsc.define_model(features_shape=features_shape, labels_shape=labels_shape)
