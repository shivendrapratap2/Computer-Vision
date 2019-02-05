
# Fruit Classifier

This Project is about designing of A CNN which would classify different fruits. 
implemented on 10,500 images of 12 different fruits.

>> Work FLow
 * creating a numpy array of labeled images and storing the data set in h5 file system. ( img shape = 100X100X3)
 * Creating an Appropriate model(Several Conv. layers Followed Pool Layers ) which would take an input of size 100X100X3 and output a 1X12    size array, which would represent probability of classification of different fruits.
 * Compiling the model.
 * Fitting the model over training set.
 * Evaluating model on test set.
 
  Dropbox Link For h5 file of data set. 
      https://www.dropbox.com/s/bz8chd4rn24v3va/data.h5
      
  Note():
   this dataset contains 4 numpy arrays: X_train_set, Y_train_set, X_test_set, Y_test_set and a integer to represent no. of classes.
