package org.deeplearning4j.examples.mlp;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.canova.api.records.reader.RecordReader;
import org.canova.api.records.reader.impl.CSVRecordReader;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/** Very simple example running Iris on Spark (local) using local data input */
public class IrisLocal {

    public static void main(String[] args) throws Exception {

        //Set up data. CSVRecordReader converts CSV data into usable format
        ClassPathResource classPathResource = new ClassPathResource("iris_shuffled_normalized_csv.txt");
        String localDataPath = "file://" + classPathResource.getURI().getPath();
        RecordReader recordReader = new CSVRecordReader(0,",");

        SparkConf sparkConf = new SparkConf();
        sparkConf.setMaster("local[*]");
        sparkConf.setAppName("Iris");

        JavaSparkContext sc = new JavaSparkContext(sparkConf);


        //Create and initialize multi-layer network
        final int numInputs = 4;
        int outputNum = 3;
        int iterations = 1;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(12345)
                .iterations(iterations)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(1e-1)
                .l1(0.01).regularization(true).l2(1e-3)
                .list(3)
                .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(3)
                        .activation("tanh")
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .layer(1, new DenseLayer.Builder().nIn(3).nOut(2)
                        .activation("tanh")
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .weightInit(WeightInit.XAVIER)
                        .activation("softmax")
                        .nIn(2).nOut(outputNum).build())
                .backprop(true).pretrain(false)
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.setUpdater(null);   //Workaround for a minor bug in 0.4-rc3.8

        //Create Spark network
        SparkDl4jMultiLayer sparkNetwork = new SparkDl4jMultiLayer(sc,net);

        int nEpochs = 6;
        List<float[]> temp = new ArrayList<>();
        for( int i=0; i<nEpochs; i++ ){
            MultiLayerNetwork network = sparkNetwork.fit(localDataPath, 4, recordReader);
            temp.add(network.params().data().asFloat().clone());
        }

        System.out.println("Parameters vs. iteration: ");
        for( int i=0; i<temp.size(); i++ ){
            System.out.println(i + "\t " + Arrays.toString(temp.get(i)));
        }
    }

}
