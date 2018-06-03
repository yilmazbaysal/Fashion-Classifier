import os
import sys

import numpy

from classifier import Classifier
from data_parser import train_dataset_parser, test_dataset_parser, train_features_parser, test_features_parser

# To close cpp warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


if sys.argv[1] == 'dataset':
    print('READING DATA FROM DATASET\n...\n')
    # train_data, binary_labels, bag_of_words = train_dataset_parser('../dataset/train/')
    test_data, image_ids = test_dataset_parser('../dataset/test/')
else:
    print('READING DATA FROM FEATURES CSV\n...\n')
    train_data, binary_labels, bag_of_words = train_features_parser('../dataset/features_train.csv')
    test_data, image_ids = test_features_parser('../dataset/features_test.csv')



#
#
#
classifier = Classifier(number_of_labels=len(binary_labels[0]))

#
#
# TRAIN
classifier.train(numpy.array(train_data), numpy.array(binary_labels))

#
#
# TEST
result = classifier.test(numpy.array(test_data))


for i in range(len(image_ids)):
    labels = []
    for j in range(len(result[i])):
        if result[i][j] >= 0.5:
            labels.append((bag_of_words[j], result[i][j]))

    print('IMAGE:', image_ids[i], 'LABELS:', labels)


print(result)

print(len(result))
print(len(result[0]))
print('MAX:', max(result[0]))


