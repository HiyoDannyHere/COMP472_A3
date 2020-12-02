import pandas as pd
import math
import source.Utilities as ut


def read_and_format_tweets(training_filepath):
    # @desc : This returns tweets as a formatted tuple.
    # @precondition : data file has a defined header (copy-paste header into offending data files if needed)
    # @param string : the name of the training data file (not full path, only the name)
    # @return [(int, [string], bool)]

    tweets = []

    for index, row in pd.read_csv(training_filepath, sep="\t", usecols=[0, 1, 2]).iterrows():
        tweets.append((int(row["tweet_id"]), row["text"].lower().split(), 1 if row["q1_label"] == "yes" else 0))
    return tweets


def gen_nb_params(tweet_list, filter_vocabulary, smoothing=0.01):
    # @desc : Will generate all relevant values for a naive bayes classifier.
    # @param string : Filename only (it is assumed to be in ./data/)
    # @param bool : Whether to filter out the 1-count words from vocabulary or not.
    # @param float : The naive bayes smoothing factor.
    # @return : ( 1, 2, 3, 4, 5 )
    #   1) {word:[yes_count, no_count]} : each word in vocabulary and its count in "yes" and "no" tweets
    #   2) int : total count of words in the "yes" tweets
    #   3) int : total count of words in the "no" tweets
    #   4) float : prior probability of a "yes" -> P(yes)
    #   5) float : prior probability of a "no" -> P(no)

    vocabulary = {}     # { word : [word_yes_count, word_no_count] }
    yes_tweet_count, no_tweet_count, yes_words_total, no_words_total, tweet_count = 0, 0, 0, 0, len(tweet_list)

    for tweet in tweet_list:
        yes_tweet_count += tweet[2] * 1             # branchless : if true: ++yes_tweet_count
        no_tweet_count += (not tweet[2]) * 1        # branchless : if false: ++no_tweet_count
        word_list = tweet[1]

        for word in word_list:
            yes_words_total += tweet[2] * 1         # branchless : if true: ++yes_words_total
            no_words_total += (not tweet[2]) * 1    # branchless : if false: ++no_words_total

            if word in vocabulary:
                vocabulary[word][0] += tweet[2] * 1
                vocabulary[word][1] += (not tweet[2]) * 1
            else:
                vocabulary[word] = [tweet[2] * 1 + smoothing, (not tweet[2]) * 1 + smoothing]

    if filter_vocabulary:   # remove all words that only appear once
        for word in vocabulary:
            if vocabulary[word][0] + vocabulary[word][1] <= 1:
                yes_words_total -= vocabulary[word][0]
                no_words_total -= vocabulary[word][1]
                del vocabulary[word]

    yes_words_total = yes_words_total + smoothing * len(vocabulary)
    no_words_total = no_words_total + smoothing * len(vocabulary)

    return vocabulary, yes_words_total, no_words_total, yes_tweet_count/tweet_count, no_tweet_count/tweet_count


def naive_bayes_classifier(tweet, nb_params):
    # @desc : This classifies "yes" and "no" and returns the actual score from the naive bayes classifier.
    # @param (tweet) : the tweet is what is returned from get_training_data()
    # @param (nb_param) : the return from gen_nb_params()
    # @return (str, float) : The classification of "yes" and "no" followed by the score.
    vocabulary, yes_words, no_words, yes_prior, no_prior = nb_params
    yes_score = math.log10(yes_prior)
    no_score = math.log10(no_prior)

    for word in tweet[1]:
        if word in vocabulary:
            yes_score += math.log10(vocabulary[word][0] / yes_words)
            no_score += math.log10(vocabulary[word][1] / no_words)

    if yes_score >= no_score:
        return "yes", yes_score
    else:
        return "no", no_score


def a3_start(training_file_name="covid_training.tsv", test_file_name="covid_test_public.tsv", smoothing=0.01):

    data_directory = "../data/"
    original_filename = "NB-BOW-OV.txt"
    filtered_filename = "NB-BOW-FV.txt"

    tr_ov, tr_fv, eval_ov, eval_fv = "", "", "", ""

    training_tweet_list = read_and_format_tweets(data_directory + training_file_name)
    test_tweet_list = read_and_format_tweets(data_directory + test_file_name)

    nb_params_ov = gen_nb_params(training_tweet_list, False, smoothing)
    nb_params_fv = gen_nb_params(training_tweet_list, True, smoothing)

    for tweet in test_tweet_list:
        answer_ov, score_ov = naive_bayes_classifier(tweet, nb_params_ov)
        answer_fv, score_fv = naive_bayes_classifier(tweet, nb_params_fv)

        tweet_answer = "yes" if tweet[2] else "no"

        is_correct_ov = "correct" if answer_ov == tweet_answer else "wrong"
        is_correct_fv = "correct" if answer_fv == tweet_answer else "wrong"

        tr_ov += str(tweet[0]) + "  " + answer_ov + "  " + "{:e}".format(score_ov) + "  " + tweet_answer + "  " \
            + is_correct_ov + "\n"
        tr_fv += str(tweet[0]) + "  " + answer_fv + "  " + "{:e}".format(score_fv) + "  " + tweet_answer + "  " \
            + is_correct_fv + "\n"

    ut.write_to_output_file("trace_" + original_filename, tr_ov)
    ut.write_to_output_file("trace_" + filtered_filename, tr_fv)
    ut.write_to_output_file("eval_" + original_filename, eval_ov)
    ut.write_to_output_file("eval_" + filtered_filename, eval_fv)
