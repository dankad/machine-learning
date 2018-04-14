package HomeWork2;

import java.util.LinkedList;
import java.util.Queue;

import weka.classifiers.Classifier;
import weka.classifiers.evaluation.output.prediction.Null;
import weka.core.*;

class Node {
    Node[] children;
    Node parent;
    int attributeIndex;
    double returnValue;
    double attributeValue; // TODO: can we do without? using this we can classify instance to his next node in the tree
    boolean done; //added by orr
    Instances data; //added by orr
}


public class DecisionTree implements Classifier {

    private Node rootNode;
    private int[] recurrences;
    private int classIndex;
    private double gini_index;
    private double entropy_index;
    private boolean usingGini = true; // TODO: decide here how should we pass this flag from the main function
    // to the buildClassifier

    @Override
    public void buildClassifier(Instances arg0) throws Exception {

        // create queue
        Queue<Node> queue = new LinkedList<>();
        //set the root node
        rootNode = new Node();
        rootNode.attributeIndex = calcGain(arg0);
        rootNode.children = createChildrenNodesByAttribute(arg0, rootNode.attributeIndex);
        rootNode.returnValue = maxClass(arg0);
        //insert all the children into the queue
        for (Node child : rootNode.children) {
            child.done = false;
            queue.add(child);
        }
        //while loop - through all nodes in the queue
        while (!queue.isEmpty()) {
            Node nodeToProcess = queue.poll();
            //if monochromatic - mark as done
            if (isMonochromatic(nodeToProcess.data)) {
                nodeToProcess.done = true;
            } else {
                //else 1.find best attribute 2.split data into children 3.for loop create children with sub data and put in queue
                nodeToProcess.attributeIndex = calcGain(arg0);
                nodeToProcess.children = createChildrenNodesByAttribute(nodeToProcess.data, nodeToProcess.attributeIndex);
                nodeToProcess.returnValue = maxClass(nodeToProcess.data);
                //insert all the children into the queue
                for (Node child : nodeToProcess.children) {
                    child.done = false;
                    queue.add(child);
                }
            }
        }
        //possible optimization add a clean up to clear all the data from the nodes
    }

    //a method that determines if a group of instances is monochromatic
    private boolean isMonochromatic(Instances data) {
        Instances instancesOfClass0 = getAllInstancesWithSameValue(data, classIndex, 0.0);
        return (instancesOfClass0.size() == 0 || instancesOfClass0.size() == data.size());
    }


    // a method that returns the class with the most instances in a given group
    private double maxClass(Instances data) {
        Instances instancesOfClass0 = getAllInstancesWithSameValue(data, classIndex, 0.0);
        Instances instancesOfClass1 = getAllInstancesWithSameValue(data, classIndex, 1.0);
        if (instancesOfClass1.size() < instancesOfClass0.size()) {
            return 0.0;
        } else {
            return 1.0;
        }
    }

    private Node[] createChildrenNodesByAttribute(Instances data, int attributeIndex) {
        // TODO not finished yet -orr
        double[] classes = data.attributeToDoubleArray(attributeIndex);

        return null;
    }

    @Override
    public double classifyInstance(Instance instance) {
        Node currNode = rootNode;
        int instHeight = 0;
        while (currNode.children[0] != null) {
            instHeight++;
            for (int i = 0; i < currNode.children.length; i++) {
                if (currNode.attributeValue == instance.toDoubleArray()[currNode.attributeIndex]) {
                    currNode = currNode.children[i];
                    break;
                }
            }
        }
        return currNode.data.instance(0).classValue();
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

    private int calcGain(Instances data) {
        double gain = 0, tmp_index, index;
        int bestAttr = 0;
        double bestGain = Integer.MIN_VALUE;
        int[] countValues;
        Instances subData;

        index = (usingGini ? getGiniIndex(data, data.classAttribute()) : getEntropyIndex(data, data.classAttribute()));

        for (int i = 0; i < data.numAttributes() - 1; i++) {
            for (int j = 0; j < data.attribute(i).numValues(); j++) {
                countValues = new int[data.attribute(i).numValues()];
                for (int k = 0; k < data.numInstances(); k++) {
                    if (data.get(k).attribute(i).toString() == data.attribute(i).value(j)) countValues[j]++;
                }
                subData = getAllInstancesWithSameValue(data, i, j);
                tmp_index = (usingGini ? getGiniIndex(subData, data.attribute(i)) : getEntropyIndex(subData, data.attribute(i)));
                gain += (countValues[j] / data.numInstances()) * tmp_index;
            }
            if (index - gain > bestGain) {
                bestGain = index - gain;
                bestAttr = i;
            }
        }
        return bestAttr;
    }

    private Instances getAllInstancesWithSameValue(Instances data, int attributeIndex, double targetValue) {

        Instances filtered = data; // TODO seems like indeed it touches the original data, let's make sure nothing important is being removed

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