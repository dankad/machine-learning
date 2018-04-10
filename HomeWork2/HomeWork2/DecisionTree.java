package HomeWork2;

import weka.classifiers.Classifier;
import weka.core.*;

class Node {
    Node[] children;
    Node parent;
    int attributeIndex;
    double returnValue;

}


public class DecisionTree implements Classifier {
    private Node rootNode;
    private int[] recurrences;
    private int classIndex;
    private double gini_index;
    private double entropy_index;

    @Override
    public void buildClassifier(Instances arg0) throws Exception {
        classIndex = arg0.classIndex();
    }

    @Override
    public double classifyInstance(Instance instance) {
        return 0;
    }

    private double getGiniIndex(Instances data, Attribute attr) {
        gini_index = 1;
        String classValue;
        recurrences = new int[attr.numValues()];
        for (int i = 0; i < data.numInstances(); i++) { // count number of instances of each class type
            for (int j = 0; j < recurrences.length; j++) {
                classValue = data.get(i).classAttribute().toString();
                if (classValue == data.classAttribute().value(j)) {
                    recurrences[j]++;
                    break;
                }
            }
        }

        for (int i = 0; i < recurrences.length; i++) { // calculate gini index for specific node
            gini_index = gini_index - Math.pow((recurrences[i] / data.numInstances()), 2);
        }
        return gini_index;
    }

    private int getGiniGain(Instances data) {
        double gini_gain = 0;
        double tmp_gini_index;
        int bestAttr = 0;
        double bestGain = Integer.MIN_VALUE;
        int[] countValues;
        Instances subData;

        gini_index = getGiniIndex(data, data.classAttribute());

        for (int i = 0; i < data.numAttributes() - 1; i++) {
            for (int j = 0; j < data.attribute(i).numValues(); j++) {
                countValues = new int[data.attribute(i).numValues()];
                for (int k = 0; k < data.numInstances(); k++) {
                    if (data.get(k).attribute(i).toString() == data.attribute(i).value(j)) countValues[j]++;
                }
                subData = getAllInstancesWithSameValue(data, i, j);
                tmp_gini_index = getGiniIndex(subData, data.attribute(i));
                gini_gain += (countValues[j] / data.numInstances()) * tmp_gini_index;
            }
            if (gini_gain > bestGain) {
                bestGain = gini_gain;
                bestAttr = i;
            }
        }
        return bestAttr;
    }

    private double getEntropyIndex(Instances data, Attribute attr) {
        entropy_index = 0;
        String classValue;
        recurrences = new int[attr.numValues()];
        for (int i = 0; i < data.numInstances(); i++) { // count number of instances of each class type
            for (int j = 0; j < recurrences.length; j++) {
                classValue = data.get(i).stringValue(classIndex);
                if (classValue == data.classAttribute().value(j)) {
                    recurrences[j]++;
                    break;
                }
            }
        }

        for (int i = 0; i < recurrences.length; i++) { // calculate gini index for specific node
            entropy_index -= (recurrences[i] / data.numInstances()) * Math.log((recurrences[i] / data.numInstances()));
        }
        return entropy_index;
    }

    private int getEntropyGain(Instances data) {
        double information_gain = 0;
        double tmp_entropy_index;
        int bestAttr = 0;
        double bestGain = Integer.MIN_VALUE;
        int[] countValues;
        Instances subData;

        entropy_index = getEntropyIndex(data, data.classAttribute());

        for (int i = 0; i < data.numAttributes() - 1; i++) {
            for (int j = 0; j < data.attribute(i).numValues(); j++) {
                countValues = new int[data.attribute(i).numValues()];
                for (int k = 0; k < data.numInstances(); k++) {
                    if (data.get(k).attribute(i).toString() == data.attribute(i).value(j)) countValues[j]++;
                }
                subData = getAllInstancesWithSameValue(data, i, j);
                tmp_entropy_index = getEntropyIndex(subData, data.attribute(i));
                information_gain += (countValues[j] / data.numInstances()) * tmp_entropy_index;
            }
            if (information_gain > bestGain) {
                bestGain = information_gain;
                bestAttr = i;
            }
        }
        return bestAttr;
    }

    private Instances getAllInstancesWithSameValue(Instances data, int attributeIndex, double targetValue) {

        Instances filtered = data;

        for (int i = 0; i < data.size(); i++) {
            if (filtered.instance(i).value(attributeIndex) != targetValue) {
                filtered.remove(i);
            }
        }
        return filtered;
    }

    @Override
    public double[] distributionForInstance(Instance arg0) throws Exception {
        // Don't change
        return null;
    }

    @Override
    public Capabilities getCapabilities() {
        // Don't change
        return null;
    }
}
