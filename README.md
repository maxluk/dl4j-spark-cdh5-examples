# DL4J Spark Examples for CDH5

Use the latest CDH5.4.2 VM cloudera to try these out:

http://www.cloudera.com/content/cloudera/en/downloads/quickstart_vms/cdh-5-4-x.html

For the general idea of getting spark jobs to run on CDH5:

http://blog.cloudera.com/blog/2014/04/how-to-run-a-simple-apache-spark-app-in-cdh-5/


Currently 3 examples are provided here:

- A LSTM RNN for predicting/generating strings of characters
- A CNN for digit classification, running on the MNIST data set
- A very simple feedforward network running on the iris data set 

More examples will be added in the future.

These examples are set up to run on Spark local (i.e., they are runnable within your IDE).
To submit and run these on a cluster, remove the .setMaster(...) method and use Spark submit. 