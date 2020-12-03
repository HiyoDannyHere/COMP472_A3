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


def gen_nb_params(tweet_list, filter_vocabulary, smoothing=0.01, filter_value=1):
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
        words_to_remove = []    # make a list of words that need filtering
        for word in vocabulary:
            if vocabulary[word][0]-smoothing + vocabulary[word][1]-smoothing <= filter_value:
                words_to_remove.append(word)
        for word in words_to_remove:    # proceed to remove those words and adjust total words accordingly
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
    # @return (bool, float) : The classification of "yes" and "no" followed by the score.
    vocabulary, yes_words, no_words, yes_prior, no_prior = nb_params
    yes_score = math.log10(yes_prior)
    no_score = math.log10(no_prior)

    for word in tweet[1]:
        if word in vocabulary:
            yes_score += math.log10(vocabulary[word][0] / yes_words)
            no_score += math.log10(vocabulary[word][1] / no_words)

    if yes_score >= no_score:
        return 1, yes_score
    else:
        return 0, no_score


def evaluation_metrics(predicted, actual, cumulative_correct_count, metrics):
    # @desc : This will fill out the metrics needed (tp, fp, tn, fn) from the "yes" and the "no" perspective. It also
    #         returns the count of correct predictions
    # @param bool : what the model predicted
    # @param bool : what the actual result is
    # @param int : pointer to an int that accumulates all the correct predictions regardless
    # @param [[int], [int]] : pointer to [[0, 0, 0, 0], [0, 0, 0, 0]]
    # @return (correct_count, [int [int x 4], [int x 4]] )
    #   ex. ( correct_count, [[tp_yes, fp_yes, tn_yes, fn_yes], [tp_no, fp_no, tn_no, fn_no]] )
    if predicted == 1:  # "yes"
        if actual == 1:
            cumulative_correct_count += 1
            metrics[0][0] += 1
            metrics[1][2] += 1
        else:
            metrics[0][1] += 1
            metrics[1][3] += 1
    else:
        if actual == 1:
            metrics[0][3] += 1
            metrics[1][1] += 1
        else:
            cumulative_correct_count += 1
            metrics[0][2] += 1
            metrics[1][0] += 1
    return cumulative_correct_count


def gen_trace_string(trace_string, tweet, predicted, score):
    # @desc : Generate the string for the trace file (format specified in A3)
    # @param string : This is the trace string that will be concatenated with successive calls to this.
    # @param (tweet) : returned by read_and_format_tweets()
    # @param bool : The predicted class : True -> yes, False -> No
    # @param float : The score calculated by naive_bayes_classifier()
    # @return string : the next line of the trace file concatenated on the previous string.
    trace_string += str(tweet[0]) + "  " + ("yes" if predicted else "no") + "  " + "{:.2E}".format(score) + "  " \
                + ("yes" if tweet[2] else "no") + "  " \
                + ("correct" if predicted == tweet[2] else "wrong") + "\n"
    return trace_string


def a3_start(training_file_name="covid_training.tsv", test_file_name="covid_test_public.tsv", smoothing=0.01):

    data_directory = "../data/"
    original_filename = "NB-BOW-OV.txt"
    filtered_filename = "NB-BOW-FV.txt"

    trace_ov, trace_fv, eval_ov, eval_fv = "", "", "", ""

    training_tweet_list = read_and_format_tweets(data_directory + training_file_name)
    test_tweet_list = read_and_format_tweets(data_directory + test_file_name)

    nb_params_ov = gen_nb_params(training_tweet_list, False, smoothing)
    nb_params_fv = gen_nb_params(training_tweet_list, True, smoothing)

    correct_ov, stats_ov = 0, [[0, 0, 0, 0], [0, 0, 0, 0]]
    correct_fv, stats_fv = 0, [[0, 0, 0, 0], [0, 0, 0, 0]]

    for tweet in test_tweet_list:
        answer_ov, score_ov = naive_bayes_classifier(tweet, nb_params_ov)
        answer_fv, score_fv = naive_bayes_classifier(tweet, nb_params_fv)

        correct_ov = evaluation_metrics(answer_ov, tweet[2], correct_ov, stats_ov)
        correct_fv = evaluation_metrics(answer_fv, tweet[2], correct_fv, stats_fv)

        trace_ov = gen_trace_string(trace_ov, tweet, answer_ov, score_ov)
        trace_fv = gen_trace_string(trace_fv, tweet, answer_fv, score_fv)

    #                       [0][0]  [0][1]  [0][2]  [0][3]    [1][0]  [1][1]  [1][2]  [1][3]
    # Evaluation metrics: [[tp_yes, fp_yes, tn_yes, fn_yes], [tp_no, fp_no, tn_no, fn_no]]
    accuracy_ov = correct_ov / len(test_tweet_list)

    precision_yes_ov = stats_ov[0][0] / (stats_ov[0][0] + stats_ov[0][1])
    precision_no_ov = stats_ov[1][0] / (stats_ov[1][0] + stats_ov[1][1])

    recall_yes_ov = stats_ov[0][0] / (stats_fv[0][0] + stats_fv[0][3])
    recall_no_ov = stats_ov[1][0] / (stats_fv[1][0] + stats_fv[1][3])

    f1_yes_ov = (2 * precision_yes_ov * recall_yes_ov)/(precision_yes_ov + recall_yes_ov)
    f1_no_ov = (2 * precision_no_ov * recall_no_ov)/(precision_no_ov + recall_no_ov)

    eval_ov = "{:0.4f}".format(accuracy_ov) + "\n" + "{:0.4f}".format(precision_yes_ov) + "  " + \
              "{:0.4f}".format(precision_no_ov) + "\n" + "{:0.4f}".format(recall_yes_ov) + "  " + \
              "{:0.4f}".format(recall_no_ov) + "\n" + "{:0.4f}".format(f1_yes_ov) + "  " + \
              "{:0.4f}".format(f1_no_ov) + "\n"

    accuracy_fv = correct_fv / len(test_tweet_list)

    precision_yes_fv = stats_fv[0][0] / (stats_fv[0][0] + stats_fv[0][1])
    precision_no_fv = stats_fv[1][0] / (stats_fv[1][0] + stats_fv[1][1])

    recall_yes_fv = stats_fv[0][0] / (stats_fv[0][0] + stats_fv[0][3])
    recall_no_fv = stats_fv[1][0] / (stats_fv[1][0] + stats_fv[1][3])

    f1_yes_fv = (2 * precision_yes_fv * recall_yes_fv)/(precision_yes_fv + recall_yes_fv)
    f1_no_fv = (2 * precision_no_fv * recall_no_fv)/(precision_no_fv + recall_no_fv)

    eval_fv = str(accuracy_fv) + "\n" + str(precision_yes_fv) + "  " + str(precision_no_fv) + "\n" + \
        str(recall_yes_fv) + "  " + str(recall_no_fv) + "\n" + str(f1_yes_fv) + "  " + str(f1_no_fv) + "\n"

    eval_fv = "{:0.4f}".format(accuracy_fv) + "\n" + "{:0.4f}".format(precision_yes_fv) + "  " + \
              "{:0.4f}".format(precision_no_fv) + "\n" + "{:0.4f}".format(recall_yes_fv) + "  " + \
              "{:0.4f}".format(recall_no_fv) + "\n" + "{:0.4f}".format(f1_yes_fv) + "  " + \
              "{:0.4f}".format(f1_no_fv) + "\n"

    ut.write_to_output_file("trace_" + original_filename, trace_ov)
    ut.write_to_output_file("trace_" + filtered_filename, trace_fv)
    ut.write_to_output_file("eval_" + original_filename, eval_ov)
    ut.write_to_output_file("eval_" + filtered_filename, eval_fv)
