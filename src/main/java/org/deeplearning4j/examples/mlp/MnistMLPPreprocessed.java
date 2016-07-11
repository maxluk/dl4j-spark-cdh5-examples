package org.deeplearning4j.examples.mlp;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParameterException;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.input.PortableDataStream;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.spark.api.Repartition;
import org.deeplearning4j.spark.api.RepartitionStrategy;
import org.deeplearning4j.spark.api.TrainingMaster;
import org.deeplearning4j.spark.api.stats.SparkTrainingStats;
import org.deeplearning4j.spark.data.DataSetExportFunction;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster;
import org.deeplearning4j.spark.stats.StatsUtils;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import scala.Tuple2;

import java.io.ByteArrayOutputStream;
import java.io.ObjectOutputStream;
import java.net.URI;
import java.util.ArrayList;
import java.util.List;

/**
 * Train a MLP on MNIST data, using preprocessed DataSet objects (saved to disk or HDFS).
 *
 * Intended to be used via Spark submit, though this can also be used locally by setting "-useSparkLocal true"
 *
 */
public class MnistMLPPreprocessed {

    @Parameter(names="-preprocessData", description = "Whether data should be saved and preprocessed (set to false to use already saved data)", arity = 1)
    private boolean preprocessData = true;

    @Parameter(names="-dataSavePath", description = "Directory in which to save the serialized data sets - required", required = true)
    private String dataSavePath;

    @Parameter(names="-useSparkLocal", description = "Use spark local (helper for testing/running without spark submit)", arity = 1)
    private boolean useSparkLocal = false;

    @Parameter(names="-batchSizePerWorker", description = "Number of examples to fit each worker with")
    private int batchSizePerWorker = 64;

    @Parameter(names="-numEpochs", description = "Number of epochs for training")
    private int numEpochs = 1;

    public static void main(String[] args) throws Exception{
        new MnistMLPPreprocessed().entryPoint(args);
    }

    protected void entryPoint(String[] args) throws Exception {
        JCommander jcmdr = new JCommander(this);
        try{
            jcmdr.parse(args);
        } catch(ParameterException e){
            //User provides invalid input -> print the usage info
            jcmdr.usage();
            try{ Thread.sleep(500); } catch(Exception e2){ }
            throw e;
        }

        SparkConf sparkConf = new SparkConf();
        if(useSparkLocal) sparkConf.setMaster("local[*]");
        sparkConf.setAppName("MLP");
        JavaSparkContext sc = new JavaSparkContext(sparkConf);

        //First: preprocess data. This isn't the most efficient approach
        if(preprocessData) {
            DataSetIterator iter = new MnistDataSetIterator(batchSizePerWorker, true, 12345);
            List<DataSet> list = new ArrayList<>();
            while (iter.hasNext()) {
                list.add(iter.next());
            }

            JavaRDD<DataSet> rdd = sc.parallelize(list);
            URI outputURI = new URI(dataSavePath);
            rdd.foreachPartition(new DataSetExportFunction(outputURI));
        }


        //----------------------------------
        //Second: conduct network training

        //Network configuration:
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(12345)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .iterations(1)
                .activation("relu")
                .weightInit(WeightInit.XAVIER)
                .learningRate(0.0069)
                .updater(Updater.NESTEROVS).momentum(0.9)
                .regularization(true).l2(1e-4)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(28*28).nOut(500).build())
                .layer(1,  new DenseLayer.Builder().nIn(500).nOut(100).build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation("softmax").nIn(100).nOut(10).build())
                .pretrain(false).backprop(true)
                .build();

        //Configuration for Spark training:
        TrainingMaster tm = new ParameterAveragingTrainingMaster.Builder(batchSizePerWorker)
                .averagingFrequency(10)
                .saveUpdater(true)
                .workerPrefetchNumBatches(2)
                .batchSizePerWorker(batchSizePerWorker)
                .repartionData(Repartition.Always)
//                .repartionData(Repartition.Never)
                .repartitionStrategy(RepartitionStrategy.Balanced)
//                .repartitionStrategy(RepartitionStrategy.SparkDefault)
                .build();

        SparkDl4jMultiLayer sparkNet = new SparkDl4jMultiLayer(sc, conf, tm);
        sparkNet.setCollectTrainingStats(true);

        //Execute training:
        for( int i=0; i<numEpochs; i++ ){
            sparkNet.fit(dataSavePath);
            System.out.println("Completed Epoch " + i);
        }

        SparkTrainingStats stats = sparkNet.getSparkTrainingStats();
        StatsUtils.exportStatsAsHtml(stats, "SparkStats.html", sc);

        System.out.println("----- DONE -----");
    }
}
