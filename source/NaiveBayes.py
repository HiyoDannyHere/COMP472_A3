import pandas as pd
from deprecated import deprecated
import math

# @todo remove default filename
def get_training_data(training_file_name="covid_training.tsv"):
    # @desc : This returns the formatted training sets. Columns are "tweet_id", "text", "q1_label"
    # @precondition : file assumed to be in the "./data/" directory relative to this source file
    # @precondition : data file has a defined header (copy-paste header into offending data files if needed)
    # @param string : the name of the training data file (not full path, only the name)
    # @return [(int, [string], bool)]

    full_training_path = "../data/" + training_file_name

    training_data = []
    for index, row in pd.read_csv(full_training_path, sep="\t", usecols=[0, 1, 2]).iterrows():
        training_data.append((int(row["tweet_id"]), row["text"].split(), 1 if row["q1_label"] == "yes" else 0))
    return training_data

# @todo implement smoothing
# @todo remove default filename
def get_naive_bayes_dataset(training_file_name="covid_training.tsv", smoothing=1):
    # @desc : Will generate all relevant values for a naive bayes classifier.
    # @param string : Filename only (it is assumed to be in ./data/)
    # @return : ( 1, 2, 3, 4, 5 )
    #   1) {word:[yes_count, no_count]} : each word in vocabulary and its count in "yes" and "no" tweets
    #   2) int : total count of words in the "yes" tweets
    #   3) int : total count of words in the "yes" tweets
    #   4) float : prior probability of a "yes"
    #   5) float : prior probability of a "no"

    training_data = get_training_data(training_file_name)
    tweet_count = len(training_data)
    vocabulary = {}     # { word : [yes_count, no_count] }
    yes_tweet_count = 0
    no_tweet_count = 0
    yes_words_total = 0
    no_words_total = 0

    for tweet in training_data:

        yes_tweet_count += tweet[2] * 1  # branchless : if true: ++yes_tweet_count
        no_tweet_count += (not tweet[2]) * 1  # branchless : if false: ++no_tweet_count
        word_list = tweet[1]

        for word in word_list:

            yes_words_total += tweet[2] * 1         # branchless : if true: ++yes_words_total
            no_words_total += (not tweet[2]) * 1    # branchless : if false: ++no_words_total

            if word in vocabulary:
                vocabulary[word][0] += tweet[2] * 1
                vocabulary[word][1] += (not tweet[2]) * 1
            else:
                vocabulary[word] = [tweet[2] * 1, (not tweet[2]) * 1]

    return vocabulary, yes_words_total, no_words_total, yes_tweet_count/tweet_count, no_tweet_count/tweet_count


def binary_naive_bayes_classifier(word_list, training_file_name="covid_training.tsv", smoothing=1):
    # @desc : This classifies a list of words (ie. a tweet) as either "yes" or "no" by Naive Bayes algorithm.
    # @param [string] : A list of words of the document to be classified
    # @return bool : True if classified as "yes", False if classified as "no"
    vocabulary, yes_words, no_words, yes_prior, no_prior = get_naive_bayes_dataset(training_file_name, smoothing)

    yes_score = math.log10(yes_prior)
    no_score = math.log10(no_prior)

    for word in word_list:
        if word in vocabulary:
            yes_score += math.log10(vocabulary[word][0] / yes_words)
            no_score += math.log10(vocabulary[word][1] / no_words)

    return yes_score >= no_score


@deprecated(reason="Daniel : old function, no longer needed but don't want to delete yet")
def get_a3_datasets(training_file_name="covid_training.tsv", test_file_name="covid_test_public.tsv"):
    # @desc : This returns the formatted training and test sets. Columns are "tweet_id", "text", "q1_label"
    # @precondition : both files are assumed to be in the "./data/" directory relative to this source file
    # @precondition : both data files have a defined header (copy-paste header into offending data files if needed)
    # @param string : the name of the training data file (not full path, only the name)
    # @param string : the name of the test data file (not full path, only the name)
    # @return ({training set}, {test set}) : { tweet_id(int) : ( [word1(str), word2(str), ...], q1_label(bool) ) }
    #   ex: 1234\tHey there!\tyes\t... => { 1234 : ( ["Hey", "there!"], 1 ) }

    full_training_path = "../data/" + training_file_name
    full_test_path = "../data/" + test_file_name

    training_data = {}
    for index, row in pd.read_csv(full_training_path, sep="\t", usecols=[0, 1, 2]).iterrows():
        training_data[int(row["tweet_id"])] = (row["text"].split(), 1 if row["q1_label"] == "yes" else 0)
    test_data = {}
    for index, row in pd.read_csv(full_test_path, sep="\t", usecols=[0, 1, 2]).iterrows():
        test_data[int(row["tweet_id"])] = (row["text"].split(), 1 if row["q1_label"] == "yes" else 0)

    return training_data, test_data


@deprecated(reason="Daniel : old function, no longer needed but don't want to delete yet")
def get_data_word_stats(data_file_tuple):
    # @desc : Will return a dictionary of the words and respective counts. Also returns the number of words (not
    #         counting doubles). Also returns the total amount of words (counting doubles). Finally, returns the
    #         number of tweets classified as "yes".
    # @instructions : use on the data returned by get_a3_datasets().
    # @param : The tuple resulting from calling get_a3_datasets().
    # @return ({str:int}, int) : ( {word(string):count(int)}, num_unique_words(int), total_words(int),
    #           number_of_yes, number_of_no)

    dictionary = {}     # dictionary of {word:count}
    word_count = 0      # total number of words in all tweets
    yes_count = 0       # how many times the tweet was classified as a "yes"

    for word_list in data_file_tuple:
        word_count += 1
        if data_file_tuple[word_list][1] == 1:
            yes_count += 1
        for word in data_file_tuple[word_list][0]:
            if word in dictionary:
                dictionary[word] += 1
            else:
                dictionary[word] = 0

    return dictionary, len(dictionary), word_count, yes_count, word_count - yes_count


@deprecated(reason="Daniel : old function, no longer needed but don't want to delete yet")
def get_vocabulary_from_word_dictionary(word_dictionary):
    # @desc :
    # @instructions :
    # @param :
    # @return :

    vocabulary = []
    for key in word_dictionary:
        vocabulary.append(key)
    vocabulary.sort()
    return vocabulary
