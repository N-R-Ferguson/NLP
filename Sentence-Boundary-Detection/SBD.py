import sys
import os
import numpy as np
from  sklearn.tree import DecisionTreeClassifier


class DecisionTree:
    def __init__(self):
        self.model = DecisionTreeClassifier(
            criterion = 'gini',
            splitter='best',
            max_depth = None,
            min_samples_split = 2,
            min_samples_leaf= 1,
            random_state=42,
        )
        
        self.y_pred = None

    def train(self, X, y) -> None:
        '''

        '''
        self.model.fit(X, y)

    def predict(self, X) -> None:
        '''

        '''
        self.y_pred = self.model.predict(X)

    def accuracy(self, y_true) -> float:
        '''

        '''
        total_correct_predictions = 0
        total_num_predictions = len(self.y_pred)

        for true, pred in zip(y_true, self.y_pred):
            if true == pred:
                total_correct_predictions += 1
        
        return ((total_correct_predictions / total_num_predictions) * 100)

    def write_to_file(self, filename, data):
        pass




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

            if left_word not in left_word_encoder:
                left_word_encoder[left_word] = index


        # RIGHT WORD
        if index != len(data)-1:
            right_word = data[index + 1, 1]
            
            if right_word not in right_word_encoder:
                right_word_encoder[right_word] = index
        # CURRENT WORD
        word_length = len(row[1])

        # L IS NUMBER
        isLNumber = left_word.isnumeric()

        # R IS CAPITALIZED
        isRCapitalized = right_word[0].isupper()

        index_array.append(row[0])
        left_word_array.append(left_word_encoder[left_word])
        right_word_array.append(right_word_encoder[right_word])
        word_length_array.append(word_length)
        isLNumber_array.append(isLNumber)
        isRCapitalized_array.append(isRCapitalized)
        classification_array.append(1 if row[2]=='EOS' else 0)

    dataframe = np.column_stack((
        np.array(index_array, dtype=np.int32), 
        np.array(left_word_array, dtype=np.int32), 
        np.array(right_word_array, dtype=np.int32), 
        np.array(word_length_array, dtype=np.int32), 
        np.array(isLNumber_array, dtype=np.bool), 
        np.array(isRCapitalized_array, dtype=np.bool), 
        np.array(classification_array, dtype=np.int32)
    ))

    return dataframe



def count_eos_neos(dataframe) -> int:
    '''
    
    '''

    neos = 0
    eos = 0

    for element in dataframe[:, -1:]:
        if element == 0: 
            neos += 1
        else: 
            eos += 1

    return eos, neos



left_word_encoder = {}
right_word_encoder = {}

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
print('Training dataset creation complete.')
test_df = create_database(test_data)
print('Test dataset creation complete.\n')



'''
    Train Dataset information:
        1) First 5 rows of data: 
            "id" | "Left Word" | "Right Word" | "Word Length" | "isLNumber" | "isRCapitalized" | "Classification
        2) Number of rows in dataset
        3) Number of EOS and NEOS entries
        4) Ratio of EOS/NEOS
'''
print('First 5 rows of Train Dataset:')
print(train_df[0:5])
print(f'{len(train_df)} total rows in Train dataset.\n')

num_eos, num_neos = count_eos_neos(train_df)
ratio = num_eos/num_neos
print(f'The Train dataset contains {num_eos} EOS and {num_neos} NEOS.',
      f'That is a ratio of {ratio:.2f} EOS/NEOS\n\n\n')




'''
    Test Dataset information:
        1) First 5 rows of data: 
            "id" | "Left Word" | "Right Word" | "Word Length" | "isLNumber" | "isRCapitalized" | "Classification
        2) Number of rows in dataset
        3) Number of EOS and NEOS entries
        4) Ratio of EOS/NEOS
'''
print('First 5 rows of Train Dataset:')
print(test_df[0:5])
print(f'{len(test_df)} total rows in Test dataset.\n')

num_eos, num_neos = count_eos_neos(test_df)
ratio = num_eos/num_neos
print(f'The Test dataset contains {num_eos} EOS and {num_neos} NEOS.',
      f'That is a ratio of {ratio:.2f} EOS/NEOS\n\n\n')




'''
TRAIN MODEL
'''
print('Declaring and initializing Decision Tree model...')
sbd_model_5_feature = DecisionTree()
print('Model creation complete.')

print('Training model inprogress...')
X_train, y_train = train_df[:, 1:-1], train_df[:, -1:]

sbd_model_5_feature.train(X_train, y_train)
print('Training complete.')






'''
TEST MODEL
'''
print('Testing inprogress...')
X_test, y_test = test_df[:, 1:-1], test_df[:, -1:]
print(X_test)
sbd_model_5_feature.predict(X_test)
sbd_model_5_feature_accuracy = sbd_model_5_feature.accuracy(y_test)
print('Testing complete.')
print(f'The Decision Tree model has an accuracy of {sbd_model_5_feature_accuracy:0.3}%')