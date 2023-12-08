# get the label image from the multichannel file
# get the label mask from the single channel files
# if a branch has more overlap with mitochondria mask, its a mitochondria (ch0)
# if a branch has more overlap with golgi mask, its a golgi (ch1)
# this is our ground truth
# train a model on all but test file with combined branch data from all files
# train a morphology only model, a motility only model, and a combined model
# test the model on the test file, have it generate a vector assigning branches to the golgi channel or mito channel (0 or 1)
# compare the vector to the ground truth for metrics
# do this for every leave one out test file --> new model for each combination
# aggregate the metrics for final results.