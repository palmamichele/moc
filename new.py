#for every model, report train and test accuracy (or loss)

#for every model, regardless of how it was trained, measure its moc.

#each network shall be initialized 5 times with a random seed, and the same seed shall be used for all models to ensure fair comparison, 
#at this regard one can report mean and std of the metrics across the 5 runs.


#notice following adv training, the loss e.g. in case of cross entropy is applied to the probabilities produced by softmax,
# moc shall follow same structure..

#cross norm (e.g. dx=dy=l_p) is something we can also try...


