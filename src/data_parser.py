import csv
import os
import re


from feature_extractor import FeatureExtractor


def train_dataset_parser(train_path):
    fe = FeatureExtractor()
    bag_of_words = set()
    images_labels = list()

    for image_name in sorted(os.listdir(train_path)):
        token_list = [int(l) for l in re.findall(r"[\d']+", image_name)]

        images_labels.append((image_name, token_list[1:]))
        bag_of_words = bag_of_words.union(set(token_list[1:]))

    #
    #
    #
    bag_of_words = list(bag_of_words)
    binary_labels = list()
    data = list()
    i = 0
    with open('../dataset/features_train.csv', 'w') as f:
        # Write bag of words the file
        f.write('{0}\n'.format(','.join([str(x) for x in bag_of_words])))

        for image_name, labels in images_labels:
            i += 1
            print(i)
            binary_vector = [0] * len(bag_of_words)

            for label in labels:
                binary_vector[bag_of_words.index(label)] = 1

            binary_labels.append(binary_vector)

            # Extract features
            spatial_features = fe.extract_features(os.path.join(train_path, image_name))
            data.append(spatial_features)

            # Write the extracted features to the file
            f.write('{0},{1}\n'.format(
                ','.join([str(x) for x in spatial_features]),
                ','.join([str(x) for x in binary_vector])
            ))

    return data, binary_labels, bag_of_words


def test_dataset_parser(test_path):
    fe = FeatureExtractor()

    data = list()
    image_ids = list()
    with open('../dataset/features_test.csv', 'w') as f:
        i = 0
        for image_name in os.listdir(test_path):
            i+=1
            print(i)
            # Extract image ID
            image_id = image_name.split('.')[0]
            image_ids.append(image_id)

            # Extract features
            spatial_features = fe.extract_features(os.path.join(test_path, image_name))
            data.append(spatial_features)

            # Write the extracted features to the file
            f.write('{0},{1}\n'.format(
                image_id,
                ','.join([str(x) for x in spatial_features]),

            ))

    return data, image_ids


def train_features_parser(train_features_path):
    spatial_features_data = list()
    binary_labels = list()

    with open(train_features_path, 'r') as f:
        bag_of_words = [int(x) for x in f.readline().split(',')]

        i = 0
        for row in csv.reader(f, delimiter=','):
            i += 1
            print(i)

            spatial_features_data.append(row[:4096])
            binary_labels.append(row[4096:])

    return spatial_features_data, binary_labels, bag_of_words


def test_features_parser(test_features_path):
    spatial_features_data = list()
    image_ids = list()

    with open(test_features_path, 'r') as f:
        for row in csv.reader(f, delimiter=','):
            image_ids.append(row[:1])
            spatial_features_data.append(row[1:])

    return spatial_features_data, image_ids
