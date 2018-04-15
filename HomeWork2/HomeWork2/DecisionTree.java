package HomeWork2;

import java.util.LinkedList;
import java.util.Queue;

import weka.classifiers.Classifier;
import weka.core.*;

class Node {
	Node[] children;
	Node parent;
	int attributeIndex;
	double returnValue;
	boolean done; //added by orr
	Instances data; //added by orr
}


public class DecisionTree implements Classifier {

	private Node rootNode;
	private int[] recurrences;
	private int classIndex;
	private double gini_index;
	private double entropy_index;
	private boolean usingGini; //added by orr

	@Override
	public void buildClassifier(Instances arg0) throws Exception {

		// create queue 
		Queue<Node> queue = new LinkedList<Node>();
		//make the root node
		Node root  = new Node();
		root.attributeIndex = getBestAttribute(arg0);
		root.children = createChildrenNodesByAtribute(root,arg0, root.attributeIndex);
		root.returnValue = maxClass(arg0);
		//insert all the children into the queue
		for(Node child : root.children){
			child.done = false;
			queue.add(child);
		}
		//while loop - through all nodes in the queue 
		while(!queue.isEmpty()){
			Node toProcces = queue.poll();
			//if moncromtic - mark as done 
			if(isMonocromatic(toProcces.data)){
				toProcces.done = true;
			}else{
				//else 1.find best atrribute 2.split data into children 3.for loop creat children with sub data and put in queue 
				toProcces.attributeIndex = getBestAttribute(toProcces.data);
				toProcces.children = createChildrenNodesByAtribute(toProcces,toProcces.data, toProcces.attributeIndex);
				toProcces.returnValue = maxClass(toProcces.data); 
				//insert all the children into the queue
				for(Node child : toProcces.children){
					child.done = false;
					queue.add(child);
				}
			}
		}
		//possible optimiztian add a clean up to clear all the data from the nodes 
	}

	//a method that determines if a group of instances is monochromatic
	private boolean isMonocromatic(Instances data) {
		Instances instancesOfclass0 = getAllInstancesWithSameValue(data, classIndex, 0.0);
		return ( instancesOfclass0.size() == 0 || instancesOfclass0.size() == data.size() ) ;
	}


	//a method that returns the class with the most instances in a given group
	private double maxClass(Instances data) {
		Instances instancesOfclass0 = getAllInstancesWithSameValue(data, classIndex, 0.0);
		Instances instancesOfclass1 = getAllInstancesWithSameValue(data, classIndex, 1.0);
		if(instancesOfclass0.size() >= instancesOfclass1.size()){
			return 0.0;
		}else{
			return 1.0;
		}
	}


	private Node[] createChildrenNodesByAtribute(Node parent,Instances data, int attributeIndex) {
		double[] classes = data.attributeToDoubleArray(attributeIndex);
		Node[] children = new Node[classes.length];
		for (int i = 0; i < classes.length; i++){
			Instances subset = getAllInstancesWithSameValue(data, attributeIndex, i);
			//create the child with the relevant subsetdata 
			Node child = new Node();
			child.data = subset;
			child.parent = parent;
			//insert to children array 
			children[i] = child; 
		}
		return children;
	}

	private int getBestAttribute(Instances data) {
		// This will use the usinggini flag to determine which to use
		int bestAtribute;
		if(this.usingGini){
			bestAtribute = getGiniGain(data);//need to be validated by dani 
		}else{
			bestAtribute = getEntropyGain(data);//need to be validated by dani
		}
		return bestAtribute;
	}

	@Override
	public double classifyInstance(Instance instance) {
		
		Node current = this.rootNode;
		while(current.children != null){
			current = current.children[(int) instance.value(current.attributeIndex)];
		}
		return current.returnValue;
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
