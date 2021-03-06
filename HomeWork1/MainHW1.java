package HomeWork1;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.concurrent.ThreadLocalRandom;

import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

public class MainHW1 {

    public static BufferedReader readDataFile(String filename) {
        BufferedReader inputReader = null;

        try {
            inputReader = new BufferedReader(new FileReader(filename));
        } catch (FileNotFoundException ex) {
            System.err.println("File not found: " + filename);
        }

        return inputReader;
    }

    /**
     * Sets the class index as the last attribute.
     *
     * @param fileName
     * @return Instances data
     * @throws IOException
     */
    public static Instances loadData(String fileName) throws IOException {
        BufferedReader datafile = readDataFile(fileName);

        Instances data = new Instances(datafile);
        data.setClassIndex(data.numAttributes() - 1);
        return data;
    }

    public static void main(String[] args) throws Exception {

        //load training data
        Instances data = loadData("wind_training.txt");

        LinearRegression test = new LinearRegression();
        test.buildClassifier(data);    //find best alpha and build classifier with all attributes

        System.out.println("The chosen alpha is: " + test.getAlpha());
        System.out.println("Training error with all features is: " + test.getError());

        //load testing data
        Instances testData = loadData("wind_testing.txt");

        LinearRegression test2 = new LinearRegression();
        test2.buildClassifier(testData);

        System.out.println("Test error with all features is: " + test2.getError());

        //build classifiers with all 3 attributes combinations
        Instances trioData = loadData("wind_training.txt");

        LinearRegression trainTrios = new LinearRegression();
        trainTrios.findBestTrio(trioData, test.getAlpha());

        //get test error for best 3 features
        Instances trioTestData = loadData("wind_testing.txt");


        LinearRegression testTrios = new LinearRegression();
        testTrios.testError(trioTestData, trainTrios.getIndexes(), test.getAlpha());

    }

}
