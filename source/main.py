import source.NaiveBayes as nb
import pandas as pd


# ['tweet_id', 'text', 'q1_label', 'q2_label', 'q3_label', 'q4_label', 'q5_label', 'q6_label', 'q7_label']

df = pd.read_csv("../data/covid_training.tsv", sep="\t")

df_a3 = df[["tweet_id", "text", "q1_label"]]

df_a3["text length"] = df_a3["text"].apply(lambda x: len(x.split()))    # add column with num words in tweet

# how many yes and no tweets
yes_no_count = df_a3[["tweet_id", "q1_label"]].groupby(["q1_label"]).count().sort_values(["q1_label"], ascending=[False])

yes_no_word_count = df_a3[["text length", "q1_label"]].groupby(["q1_label"]).mean().sort_values(["text length"], ascending=[False])

print(yes_no_word_count)
