package org.deeplearning4j.examples.stats;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.storage.StorageLevel;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.examples.rnn.GravesLSTMCharModellingExample;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.layers.setup.ConvolutionLayerSetup;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.spark.api.Repartition;
import org.deeplearning4j.spark.api.stats.SparkTrainingStats;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster;
import org.deeplearning4j.spark.stats.EventStats;
import org.deeplearning4j.spark.stats.StatsUtils;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;

/**
 * This example is designed to show how to use DL4J's Spark training benchmarking/debugging/timing functionality.
 *
 * The idea with this tool is to capture statistics on various aspects of Spark training, in order to identify
 * and debug performance issues.
 *
 * For the sake of the example, we will be using a network configuration and data as per the RNN example
 * @author Alex Black
 */
public class TrainingStatsExample {
    private static final Logger log = LoggerFactory.getLogger(TrainingStatsExample.class);

    public static void main(String[] args) throws Exception {
        //Set up network configuration:
        MultiLayerNetwork net = new MultiLayerNetwork(GravesLSTMCharModellingExample.getConfiguration());
        net.init();

        //Set up the Spark-specific configuration
        int examplesPerWorker = 8;
        int averagingFrequency = 3;
        int nWorkers = 6;

        //Set up Spark configuration and context
        SparkConf sparkConf = new SparkConf();
        sparkConf.setMaster("local[" + nWorkers + "]");
        sparkConf.setAppName("LSTM_Char");
        JavaSparkContext sc = new JavaSparkContext(sparkConf);

        //Get data. See GravesLSTMCharModellingExample for details
        List<String> list = GravesLSTMCharModellingExample.getShakespeareAsList(GravesLSTMCharModellingExample.sequenceLength);
        JavaRDD<String> rawStrings = sc.parallelize(list);
        rawStrings.persist(StorageLevel.MEMORY_ONLY());
        final Broadcast<Map<Character, Integer>> bcCharToInt = sc.broadcast(GravesLSTMCharModellingExample.CHAR_TO_INT);
        JavaRDD<DataSet> data = rawStrings.map(new GravesLSTMCharModellingExample.StringToDataSetFn(bcCharToInt));

        //Set up the TrainingMaster. The TrainingMaster controls how learning is actually executed on Spark
        //Here, we are using standard parameter averaging
        int examplesPerDataSetObject = 1;
        ParameterAveragingTrainingMaster tm = new ParameterAveragingTrainingMaster.Builder(examplesPerDataSetObject)
                .repartionData(Repartition.NumPartitionsWorkersDiffers)
                .workerPrefetchNumBatches(0)    //Don't asynchronously prefetch batches
                .saveUpdater(true)
                .averagingFrequency(averagingFrequency)
                .batchSizePerWorker(examplesPerWorker)
                .build();


        SparkDl4jMultiLayer sparkNetwork = new SparkDl4jMultiLayer(sc, net, tm);

        //*** Tell the network to collect training statistics. These will NOT be collected by default ***
        sparkNetwork.setCollectTrainingStats(true);

        //Fit for 1 epoch:
        sparkNetwork.fit(data);

        //Get the statistics:
        SparkTrainingStats stats = sparkNetwork.getSparkTrainingStats();
        Set<String> statsKeySet = stats.getKeySet();    //Keys for the types of statistics
        System.out.println("--- Collected Statistics ---");
        for(String s : statsKeySet){
            System.out.println(s);
        }

        //Demo purposes: get one statistic
        String first = statsKeySet.iterator().next();
        List<EventStats> firstStatEvents = stats.getValue(first);
        EventStats es = firstStatEvents.get(0);
        System.out.println("Machine ID\t" + es.getMachineID());
        System.out.println("JVM ID\t" + es.getJvmID());
        System.out.println("Thread ID\t" + es.getThreadID());
        System.out.println("Start time ms\t" + es.getStartTime());
        System.out.println("Duration ms\t" + es.getDurationMs());


        //Print out statistics as a String representation
        System.out.println(stats.statsAsString());

        //Export a HTML file containing charts of the various stats calculated during training
        StatsUtils.exportStatsAsHtml(stats, "SparkStats.html",sc);

        log.info("****************Example finished********************");
    }
}
