# Myers-Briggs-PersTypes

A Naive-Bayes approach to personality type prediction using Python.

This is a supervised learning task, with the data being 287,000 comments written by individuals on the 16Personalities website, labeled by the individual's personality type https://www.kaggle.com/datasnaek/mbti-type.

The dataset is very unbalanced, so the weight of each sample is the inverse of its presence in the dataset. 

Text was first preprocessed to remove unimportant features, then vectorised using the TfidfVectoriser from Sklearn.

Initially, a 16 output model was chosen, one for each personality type, however there was not enough information to differentiate between all 16 types and resulted in a small subset of the sample space being predicted at all. Instead, a model was trained on each dimension [IE, SN, TF, JP], which showed better predictive ability (86% accuracy on SN dimension). 

The model though only predicts somewhat better than average (i.e. what would be obtained from just predicting the priors), and so more must be done to reject the hypothesis that there is not enough information present in the words individuals use to predict their Myers-Briggs personality type.
