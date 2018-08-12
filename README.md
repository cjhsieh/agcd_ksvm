Accelerated Asynchronous Parallel Kernel SVM 
========================


Build
---------------

We require the following environment to build Asyn-KSVM:

- Unix Systems (We haven't tested on Mac OS). 

To build the program, simply run `make`. Two binaries, `svm-train` (for training) 
and `svm-predict` (for prediction) will be built.  

Data Preparation 
----------------

Please download the datasets from LIBSVM datasets
http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets

We already have the ijcnn1.tr (training) and ijcnn1.t (testing) set
in the folder. 

Usage
----------------

./svm-train is the training procedure and it can print out the prediction 
accuracy at each step.

Type ./svm-train to get the usage: 

```
Usage: svm-train [options] training_set_file testing_set_file [model_file]
options:
-s svm_type : set type of SVM (default 0)
        0 -- C-SVC              (multi-class classification)
-t kernel_type : set type of kernel function (default 2)
        0 -- linear: u'*v
		1 -- polynomial: (gamma*u'*v + coef0)^degree
		2 -- radial basis function: exp(-gamma*|u-v|^2)
		3 -- sigmoid: tanh(gamma*u'*v + coef0)
-d degree : set degree in kernel function (default 3)
-g gamma : set gamma in kernel function (default 1/num_features)
-r coef0 : set coef0 in kernel function (default 0)
-c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)
-m cachesize : set total cache memory size in MB (default 2000); each thread will use cachesize/n memory
-e epsilon : set tolerance of termination criterion (default 0.001)
-wi weight : set the parameter C of class i to weight*C, for C-SVC (default 1)
-v n: n-fold cross validation mode
-T maxiter : Maximum number of iterations (default 10*(number of samples))
===========
Parameters for accelerated greedy coordinate descent: 
-u mu: set mu parameter for Accelerated Greedy CD
-k k: set k parameter for Accelerated Greedy CD (called t in matlab implementation)
-a accelerated_type:
	0 -- no acceleration
	1 -- accelerated greedy cd (strongly convex version)
```


./svm-predict is used for prediction: 
```
Usage: svm-predict [options] test_file model_file output_file
options:
-b probability_estimates: whether to predict probability estimates, 0 or 1 (default 0)
-q : quiet mode (no outputs)
```

Examples: 
------------------------------

Run 100000 iterations on ijcnn1.tr (testing on ijcnn1.t), with strongly convex accleration
```
./svm-train -a 1 -c 32 -g 2 -T 100000 ijcnn1.tr ijcnn1.t
```

Run 100000 iterations of original svm (no-bias term version)
```
./svm-train -a 0 -c 32 -g 2 -T 100000 ijcnn1.tr ijcnn1.t
```

