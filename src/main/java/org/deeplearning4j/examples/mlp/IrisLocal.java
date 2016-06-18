package org.deeplearning4j.examples.mlp;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.canova.api.records.reader.RecordReader;
import org.canova.api.records.reader.impl.CSVRecordReader;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.FeedForwardLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.spark.canova.RecordReaderFunction;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/** Very simple example running Iris on Spark (local) using local data input */
public class IrisLocal {

    public static void main(String[] args) throws Exception {
        int nWorkers = 5;
        SparkConf sparkConf = new SparkConf();
        sparkConf.setMaster("local[" + nWorkers + "]");
        sparkConf.setAppName("Iris");

        JavaSparkContext sc = new JavaSparkContext(sparkConf);

        //Load the data from local (driver) classpath into a JavaRDD<DataSet>, for training
            //CSVRecordReader converts CSV data (as a String) into usable format for network training
        RecordReader recordReader = new CSVRecordReader(0,",");
        File f = new File("src/main/resources/iris_shuffled_normalized_csv.txt");
        JavaRDD<String> irisDataLines = sc.textFile(f.getAbsolutePath());
        int labelIndex = 4;
        int numOutputClasses = 3;
        JavaRDD<DataSet> trainingData = irisDataLines.map(new RecordReaderFunction(recordReader, labelIndex, numOutputClasses));


        //First: Create and initialize multi-layer network. Configuration is the same as in normal (non-distributed) DL4J training
        final int numInputs = 4;
        int outputNum = 3;
        int iterations = 1;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(12345)
                .iterations(iterations)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(0.5)
                .regularization(true).l2(1e-4)
                .activation("tanh")
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(3).build())
                .layer(1, new DenseLayer.Builder().nIn(3).nOut(2).build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT).activation("softmax")
                        .nIn(2).nOut(outputNum).build())
                .backprop(true).pretrain(false)
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();



        //Second: Set up the Spark training.
        //Set up the TrainingMaster. The TrainingMaster controls how learning is actually executed on Spark
        //Here, we are using standard parameter averaging
        ParameterAveragingTrainingMaster tm = new ParameterAveragingTrainingMaster.Builder(nWorkers)
                .workerPrefetchNumBatches(2)    //Asynchronously prefetch up to 2 batches
                .saveUpdater(true)
                .averagingFrequency(1)  //See comments on averaging frequency in LSTM example. Averaging every 1 iteration is inefficient in practical problems
                .batchSizePerWorker(30)
                .build();
        SparkDl4jMultiLayer sparkNetwork = new SparkDl4jMultiLayer(sc,net,tm);

        int nEpochs = 100;
        for( int i=0; i<nEpochs; i++ ){
            sparkNetwork.fit(trainingData);
        }


        //Finally: evaluate the (training) data accuracy in a distributed manner:
        Evaluation evaluation = sparkNetwork.evaluate(trainingData);
        System.out.println(evaluation.stats());
    }

}
