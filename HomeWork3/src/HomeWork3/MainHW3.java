package HomeWork3;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

import weka.core.Instances;

public class MainHW3 {

    public static BufferedReader readDataFile(String filename) {
        BufferedReader inputReader = null;

        try {
            inputReader = new BufferedReader(new FileReader(filename));
        } catch (FileNotFoundException ex) {
            System.err.println("File not found: " + filename);
        }

        return inputReader;
    }

    public static Instances loadData(String fileName) throws IOException {
        BufferedReader datafile = readDataFile(fileName);
        Instances data = new Instances(datafile);
        data.setClassIndex(data.numAttributes() - 1);
        return data;
    }

    public static void main(String[] args) throws Exception {

        Instances trainingAutoPrice = loadData("auto_price.txt");
        FeatureScaler featureScaler = new FeatureScaler();
        Instances scaledTrainingAutoPrice = featureScaler.scaleData(trainingAutoPrice);
        Knn knn;
        double tempValError, valError = Integer.MAX_VALUE;
        int m_k = 0, lp = 0, maj = 0;
        int[] numOfFolds = {trainingAutoPrice.size(), 50, 10, 5, 3};

        for (int i = 1; i <= 20; i++) {
            for (int j = 1; j <= 4; j++) {
                if (j == 4) {
                    knn = new Knn(false, 1, true, i, false);
                    tempValError = knn.crossValidationError(trainingAutoPrice, 10);
                    System.out.println("Current validation error is " + tempValError + " in K: " + i);
                    if (tempValError < valError) {
                        valError = tempValError;
                        m_k = i;
                        lp = 4;
                        maj = 0;
                    }
                } else {
                    knn = new Knn(false, j, false, i, false);
                    tempValError = knn.crossValidationError(trainingAutoPrice, 10);
                    System.out.println("Current validation error is " + tempValError + " in K: " + i);
                    if (tempValError < valError) {
                        valError = tempValError;
                        m_k = i;
                        lp = j;
                        maj = 0;
                    }
                }

            }
        }


        // uniform check of original data
//        for (int i = 1; i <= 20; i++) {
//            for (int j = 1; j <= 4; j++) {
//                for (int k = 0; k < numOfFolds.length; k++) {
//                    if (j == 4) {
//                        knn = new Knn(false, 1, true, i, false);
//                        tempValError = knn.crossValidationError(trainingAutoPrice, numOfFolds[k]);
//                        if (tempValError < valError) {
//                            valError = tempValError;
//                            m_k = i;
//                            lp = 4;
//                            maj = 0;
//                        }
//                    } else {
//                        knn = new Knn(false, j, false, i, false);
//                        tempValError = knn.crossValidationError(trainingAutoPrice, numOfFolds[k]);
//                        if (tempValError < valError) {
//                            valError = tempValError;
//                            m_k = i;
//                            lp = j;
//                            maj = 0;
//                        }
//                    }
//                }
//            }
//        }
//
//        // weighted check of original data
//        for (int i = 1; i <= 20; i++) {
//            for (int j = 1; j <= 4; j++) {
//                for (int k = 0; k < numOfFolds.length; k++) {
//                    if (j == 4) {
//                        knn = new Knn(true, 1, true, i, false);
//                        tempValError = knn.crossValidationError(trainingAutoPrice, numOfFolds[k]);
//                        if (tempValError < valError) {
//                            valError = tempValError;
//                            m_k = i;
//                            lp = 4;
//                            maj = 1;
//                        }
//                    } else {
//                        knn = new Knn(true, j, false, i, false);
//                        tempValError = knn.crossValidationError(trainingAutoPrice, numOfFolds[k]);
//                        if (tempValError < valError) {
//                            valError = tempValError;
//                            m_k = i;
//                            lp = j;
//                            maj = 1;
//                        }
//                    }
//                }
//            }
//        }
//
//        // uniform check of scaled data
//        for (int i = 1; i <= 20; i++) {
//            for (int j = 1; j <= 4; j++) {
//                for (int k = 0; k < numOfFolds.length; k++) {
//                    if (j == 4) {
//                        knn = new Knn(false, 1, true, i, false);
//                        tempValError = knn.crossValidationError(scaledTrainingAutoPrice, numOfFolds[k]);
//                        if (tempValError < valError) {
//                            valError = tempValError;
//                            m_k = i;
//                            lp = 4;
//                            maj = 0;
//                        }
//                    } else {
//                        knn = new Knn(false, j, false, i, false);
//                        tempValError = knn.crossValidationError(scaledTrainingAutoPrice, numOfFolds[k]);
//                        if (tempValError < valError) {
//                            valError = tempValError;
//                            m_k = i;
//                            lp = j;
//                            maj = 0;
//                        }
//                    }
//                }
//            }
//        }
//
//        // weighted check of scaled data
//        for (int i = 1; i <= 20; i++) {
//            for (int j = 1; j <= 4; j++) {
//                for (int k = 0; k < numOfFolds.length; k++) {
//                    if (j == 4) {
//                        knn = new Knn(false, 1, true, i, false);
//                        tempValError = knn.crossValidationError(trainingAutoPrice, numOfFolds[k]);
//                        if (tempValError < valError) {
//                            valError = tempValError;
//                            m_k = i;
//                            lp = 4;
//                            maj = 1;
//                        }
//                    } else {
//                        knn = new Knn(false, j, false, i, false);
//                        tempValError = knn.crossValidationError(trainingAutoPrice, numOfFolds[k]);
//                        if (tempValError < valError) {
//                            valError = tempValError;
//                            m_k = i;
//                            lp = j;
//                            maj = 1;
//                        }
//                    }
//                }
//            }
//        }
        System.out.println("Cross validation error with K = " + m_k +
                ", lp = " + lp + ", majority function = " + maj + " for auto_price data is: " + valError);
    }
}
