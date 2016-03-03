package org.deeplearning4j.examples.cnn;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.storage.StorageLevel;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
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
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;

/**Simple example of learning MNIST with spark (local)
 * NOTE: This example runs and gives reasonable results, but better performance could be obtained
 * with some additional tuning of network hyperparameters
 * @author Alex Black
 */
public class MnistExample {
    private static final Logger log = LoggerFactory.getLogger(MnistExample.class);

    public static void main(String[] args) throws Exception {

        //Create spark context
        int nCores = 6; //Number of CPU cores to use for training
        SparkConf sparkConf = new SparkConf();
        sparkConf.setMaster("local[" + nCores + "]");
        sparkConf.setAppName("MNIST");
        sparkConf.set(SparkDl4jMultiLayer.AVERAGE_EACH_ITERATION, String.valueOf(true));
        JavaSparkContext sc = new JavaSparkContext(sparkConf);


        int nChannels = 1;
        int outputNum = 10;
        int numSamples = 60000;
        int nTrain = 50000;
        int nTest = 10000;
        int batchSize = 50;
        int iterations = 1;
        int seed = 123;

        //Load data into memory
        log.info("Load data....");
        DataSetIterator mnistIter = new MnistDataSetIterator(1, numSamples, true);
        List<DataSet> allData = new ArrayList<>(numSamples);
        while(mnistIter.hasNext()){
            allData.add(mnistIter.next());
        }
        Collections.shuffle(allData,new Random(12345));

        Iterator<DataSet> iter = allData.iterator();
        List<DataSet> train = new ArrayList<>(nTrain);
        List<DataSet> test = new ArrayList<>(nTest);

        int c = 0;
        while(iter.hasNext()){
            if(c++ <= nTrain) train.add(iter.next());
            else test.add(iter.next());
        }

        JavaRDD<DataSet> sparkDataTrain = sc.parallelize(train);
        sparkDataTrain.persist(StorageLevel.MEMORY_ONLY());

        //Set up network configuration
        log.info("Build model....");
        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .regularization(true).l2(0.0005)
                .learningRate(0.1)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.ADAGRAD)
                .list(6)
                .layer(0, new ConvolutionLayer.Builder(5, 5)
                        .nIn(nChannels)
                        .stride(1, 1)
                        .nOut(20)
                        .weightInit(WeightInit.XAVIER)
                        .activation("relu")
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{2, 2})
                        .build())
                .layer(2, new ConvolutionLayer.Builder(5, 5)
                        .nIn(20)
                        .nOut(50)
                        .stride(2,2)
                        .weightInit(WeightInit.XAVIER)
                        .activation("relu")
                        .build())
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{2, 2})
                        .build())
                .layer(4, new DenseLayer.Builder().activation("relu")
                        .weightInit(WeightInit.XAVIER)
                        .nOut(200).build())
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNum)
                        .weightInit(WeightInit.XAVIER)
                        .activation("softmax")
                        .build())
                .backprop(true).pretrain(false);
        new ConvolutionLayerSetup(builder,28,28,1);

        MultiLayerConfiguration conf = builder.build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.setUpdater(null);   //Workaround for minor bug in 0.4-rc3.8

        //Create Spark multi layer network from configuration
        SparkDl4jMultiLayer sparkNetwork = new SparkDl4jMultiLayer(sc, net);

        //Train network
        log.info("--- Starting network training ---");
        int nEpochs = 5;
        for( int i=0; i<nEpochs; i++ ){
            //Run learning. Here, we are training with approximately 'batchSize' examples on each executor
            net = sparkNetwork.fitDataSet(sparkDataTrain, nCores * batchSize);
            System.out.println("----- Epoch " + i + " complete -----");

            //Evaluate (locally)
            Evaluation eval = new Evaluation();
            for(DataSet ds : test){
                INDArray output = net.output(ds.getFeatureMatrix());
                eval.eval(ds.getLabels(),output);
            }
            log.info(eval.stats());
        }

        log.info("****************Example finished********************");
    }
}
