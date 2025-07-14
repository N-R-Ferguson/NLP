import sys
import os
import numpy as np
import sklearn.tree as Tree


def create_database(data) -> np.array:
    '''
        Left Word
        Right Word
        Length of Word
        L is Number
        R is Capitalized
    '''
    index_array = []
    left_word_array = []
    right_word_array = []
    word_length_array = []
    isLNumber_array = []
    isRCapitalized_array = []
    classification_array = []

    left_word = ''
    right_word=''

    for index, row in enumerate(data):
        if (row[2] == 'TOK'):
            continue

        # LEFT WORD
        if index != 0:
            left_word = data[index - 1, 1]

        # RIGHT WORD
        if index != len(data)-1:
            right_word = data[index + 1, 1]
            
        # CURRENT WORD
        word_length = len(row[1])

        # L IS NUMBER
        isLNumber = left_word.isnumeric()

        # R IS CAPITALIZED
        isRCapitalized = right_word[0].isupper()

        index_array.append(row[0])
        left_word_array.append(left_word)
        right_word_array.append(right_word)
        word_length_array.append(word_length)
        isLNumber_array.append(isLNumber)
        isRCapitalized_array.append(isRCapitalized)
        classification_array.append(row[2])

    dataframe = np.column_stack((
        np.array(index_array, dtype=np.int32), 
        np.array(left_word_array, dtype='U25'), 
        np.array(right_word_array, dtype='U25'), 
        np.array(word_length_array, dtype=np.int32), 
        np.array(isLNumber_array, dtype=np.bool), 
        np.array(isRCapitalized_array, dtype=np.bool), 
        np.array(classification_array, dtype='U4')
    ))

    return dataframe


DATA_DIR = "NLP-Datasets/SBD/"

train_filename = sys.argv[1]
test_filename = sys.argv[2]

train_filepath = os.path.join(DATA_DIR, train_filename)
test_filepath = os.path.join(DATA_DIR, test_filename)

try:
    print(f'Reading {train_filename} and {test_filename}...')
    '''
        Files contain the # character which are read as the start of a comment 
        by NumPy's loadtxt and genfromtxt functions.
        The comments parameter in loadtxt should be set to something other 
    '''
    train_data = np.loadtxt(fname=train_filepath, delimiter=' ', dtype=np.str_, comments="////")
    test_data = np.loadtxt(fname=test_filepath, delimiter=' ', dtype=np.str_, comments="////")
except ValueError as e: 
    print(e)
except FileNotFoundError as e:
    print(e)
finally:
    print("Finished reading in data from: ", train_filename)
    print("Finished reading in data from: ", test_filename, '\n')




print('Creating train and test dataframes...')
train_df = create_database(train_data)
print('Training dataset creation complete')
test_df = create_database(test_data)
print('Test dataset creation complete\n')

print('First 5 rows of Train Dataset:')
print(train_df[0:5])
print(f'{len(train_df)} total rows in Train dataset\n')

print('First 5 rows of Train Dataset:')
print(test_df[0:5])
print(f'{len(test_df)} total rows in Test dataset\n')

