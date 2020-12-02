import source.NaiveBayes as nb

vocabulary, yes_word_count, no_word_count, yes_prior, no_prior = nb.get_naive_bayes_dataset()
print(vocabulary)
print(yes_prior + no_prior)


