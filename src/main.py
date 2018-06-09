import os
import sys

import numpy
from matplotlib import pyplot

from classifier import Classifier
from data_parser import train_dataset_parser, test_dataset_parser, train_features_parser, test_features_parser

# To close cpp warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


if sys.argv[1] == 'dataset':
    print('READING DATA FROM DATASET\n...\n')
    train_data, binary_labels, bag_of_words = train_dataset_parser('../dataset/train/')
    test_data, image_ids = test_dataset_parser('../dataset/test/')
else:
    print('READING DATA FROM FEATURES CSV\n...\n')
    train_data, binary_labels, bag_of_words = train_features_parser('../dataset/features_train.csv')


#########
# TRAIN #
#########

# Convert train data to a numpy array
train_data = numpy.array(train_data)
binary_labels = numpy.array(binary_labels)

print(len(train_data))

classifier = Classifier(number_of_labels=len(binary_labels[0]))

# Train the model
metrics = classifier.train(train_data, binary_labels)

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

#
#
# Plot the results
pyplot.figure('Train metrics')

pyplot.subplot(221)
pyplot.plot(metrics.losses)
pyplot.ylabel('Train Loss')

pyplot.subplot(222)
pyplot.plot(metrics.val_losses)
pyplot.ylabel('Validation Loss')

pyplot.subplot(223)
pyplot.plot(metrics.accuracies)
pyplot.ylabel('Train Accuracy')

pyplot.subplot(224)
pyplot.plot(metrics.val_accuracies)
pyplot.ylabel('Validation Accuracy')

pyplot.show()

