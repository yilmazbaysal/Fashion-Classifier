import os
import sys

import numpy

from classifier import Classifier
from data_parser import train_dataset_parser, test_dataset_parser, train_features_parser, test_features_parser

# To close cpp warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


if sys.argv[1] == 'dataset':
    print('READING DATA FROM DATASET\n...\n')
    train_data, binary_labels, bag_of_words = train_dataset_parser('../dataset/train/')
    # test_data, image_ids = test_dataset_parser('../dataset/test/')
else:
    print('READING DATA FROM FEATURES CSV\n...\n')
    train_data, binary_labels, bag_of_words = train_features_parser('../dataset/features_train.csv')
    # test_data, image_ids = test_features_parser('../dataset/features_test.csv')

#
#
# Create a classifier
# classifier = Classifier(number_of_labels=len(binary_labels[0]))


#########
# TRAIN #
#########

# Convert train data to a numpy array
train_data = numpy.array(train_data)
binary_labels = numpy.array(binary_labels)

print(len(train_data))

classifier = Classifier(number_of_labels=len(binary_labels[0]))

# Train the model
classifier.train(train_data, binary_labels)


########
# TEST #
########

# Parse the test data
if sys.argv[1] != 'dataset':
    test_data, image_ids = test_features_parser('../dataset/features_test.csv')
else:
    test_data, image_ids = ([], [])

# Test the model
result = classifier.test(numpy.array(test_data))


# Print results
with open('../dataset/output.csv', 'w') as f:
    f.write('image,label\n')
    for i in range(len(image_ids)):
        labels = set()
        max_probability = (0, 0)
        for j in range(len(result[i])):
            # Find the max probability
            if result[i][j] > max_probability[1]:
                max_probability = (bag_of_words[j], result[i][j])

            # Get probabilities higher than a threshold
            if result[i][j] >= 0.65:
                labels.add((bag_of_words[j], result[i][j]))

        # Add the max probability to the set (It is set, because max probability can be already exists in it)
        labels.add(max_probability)

        f.write('{0},{1}\n'.format(int(image_ids[i][0]), ' '.join([str(l[0]) for l in labels])))
        print('IMAGE:', image_ids[i][0], 'LABELS:', labels)

print(result)
print(len(result))
print(len(result[0]))
print('MAX:', max(result[0]))


