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

	public Node rootNode;
	private int[] recurrences; //what is this for????
	private int classIndex;
	private double gini_index;
	private double entropy_index;
	public boolean usingGini; //added by orr
	public int maxHightOfTree;
	public double pValue;

	private final double[][] chiSqureTable = {
			//the max degree in this data is 11 attribute values 
			//and only the relevant p values  
			//p: 1,  0.75,   0.5,  0.25,  0.05, 0.005
			{0, 0, 0, 0, 0, 0},                        // deg 0
			{0, 0.102, 0.455, 1.323, 3.841, 7.879},    // deg 1
			{0, 0.575, 1.386, 2.773, 5.991, 10.597},   // deg 2
			{0, 1.213, 2.366, 4.108, 7.815, 12.838},   // deg 3
			{0, 1.923, 3.357, 5.385, 9.488, 14.860},   // deg 4
			{0, 2.675, 4.351, 6.626, 11.070, 16.750},  // deg 5
			{0, 3.455, 5.348, 7.841, 12.592, 18.548},  // deg 6
			{0, 4.255, 6.346, 9.037, 14.067, 20.278},  // deg 7
			{0, 5.071, 7.344, 10.219, 15.507, 21.955}, // deg 8
			{0, 5.899, 8.343, 11.389, 16.919, 23.589}, // deg 9
			{0, 6.737, 9.342, 12.549, 18.307, 25.188}, // deg 10
			{0, 7.584, 10.341, 13.701, 19.675, 26.757} // deg 11
	};


	@Override
	public void buildClassifier(Instances arg0) throws Exception {

		// create queue 
		Queue<Node> queue = new LinkedList<Node>();
		//make the root node
		this.rootNode  = new Node();
		rootNode.data = arg0;
		rootNode.attributeIndex = getBestAttribute(arg0);
		rootNode.children = createChildrenNodesByAtribute(rootNode ,arg0, rootNode.attributeIndex);
		rootNode.returnValue = maxClass(arg0);
		//insert all the children into the queue
		for(Node child : rootNode.children){
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
				if(!toPrun(toProcces.data , toProcces.attributeIndex)){
					toProcces.children = createChildrenNodesByAtribute(toProcces ,toProcces.data, toProcces.attributeIndex);
					toProcces.returnValue = maxClass(toProcces.data); 
					//insert all the children into the queue
					for(Node child : toProcces.children){
						child.done = false;
						queue.add(child);
					}
				}else{
					toProcces.done = true;//prune the Tree at this node
				}
			}
		}
		//possible optimiztian add a clean up to clear all the data from the nodes 
	}


	//a warper method for the pruning decision process
	private boolean toPrun(Instances data, int attributeIndex) {
		return(calcChiSquare(data, attributeIndex) < getChiTableValue(data , attributeIndex));
	}

	private double calcChiSquare(Instances data, int attributeIndex) {
		//--------------------------not orignal code - only for testing------------------- 
		int df = 0, pf = 0, nf = 0;
    	double p0 = 0, p1 = 0, e0 = 0, e1 = 0, chiSquareStat = 0;
    	int numOfValues = data.attribute(attributeIndex).numValues(); // the range of values in a specific attribute
    	
		for (int i = 0; i < data.size(); i++)
		{
			if (data.instance(i).classValue() == 0)
				p0++;
			else p1++;	
		}
		
		p0 = p0 / data.size();
		p1 = p1 / data.size();
		
    	for (int attributeVal = 0; attributeVal < numOfValues; attributeVal++) 
    	{
			
    		for (int j = 0; j < data.size(); j++)
    		{
    			if(data.instance(j).value(attributeIndex) == attributeVal)
    			{
    				df++;
    				if (data.instance(j).classValue() == 0)
    					pf++;
    				else	nf++;
    			}
    		}
    		
    		e0 = df * p0;
    		e1 = df * p1;
    		
    		//check if this attributeVal exist in the data
    		if (df != 0)
    		{
    			chiSquareStat += (Math.pow(pf-e0, 2) / e0) + (Math.pow(nf-e1, 2) / e1); 
    		}
    		
    		df = 0;
    		pf = 0;
    		nf = 0;
		}
    	
    	return chiSquareStat;

	}

//---------------------------------------end of not orignal code -------------------------------------
	private double getChiTableValue(Instances data, int attributeIndex) {


		int degreeOfFreedom = 0;
		for (int i = 0; i < data.attribute(attributeIndex).numValues(); i++) {
			if(getAllInstancesWithSameValue(data, attributeIndex, (double)i).size() != 0)
				degreeOfFreedom++;
		}

		//converet pValue into the right colom of the table 
		int pValueIndex = 0;

		//posiible to move to enum or swich case 
		if(pValue == 0.005)
			pValueIndex = 5;
		else if(pValue == 0.05)
			pValueIndex = 4;
		else if(pValue == 0.25)
			pValueIndex = 3;
		else if(pValue == 0.5)
			pValueIndex = 2;
		else if(pValue == 0.75)
			pValueIndex = 1;
		else if(pValue == 1)
			pValueIndex = 0;

		return chiSqureTable[degreeOfFreedom - 1][pValueIndex];

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

    // a method that returns an array of Nodes by atrirbute  
	private Node[] createChildrenNodesByAtribute(Node parent,Instances data, int attributeIndex) {
		
		double[] classes = data.attributeToDoubleArray(attributeIndex);
		Node[] children = new Node[data.attribute(attributeIndex).numValues()];
		for (int i = 0; i < children.length; i++){
			Instances subset = getAllInstancesWithSameValue(data, attributeIndex, i);
			double[] test = subset.attributeToDoubleArray(attributeIndex);
			//create the child with the relevant subsetdata 
			Node child = new Node();
			child.data = subset;
			child.parent = parent;
			//insert to children array 
			children[i] = child; 
		}
		return children;
	}
	
	
	//a method that fillters instances by value at a given attribute
	// tested 
	public Instances getAllInstancesWithSameValue(Instances data, int attributeIndex, double targetValue) {

		for (int i = (data.size() - 1 ); i >= 0; i--) {
			if (data.instance(i).value(attributeIndex) != targetValue) {
				data.remove(data.get(i));
			}
		}
		return data;
	}

	
	/*
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
	*/

	@Override
	public double classifyInstance(Instance instance) {

		Node current = this.rootNode;
		while(current.children != null){
			current = current.children[(int) instance.value(current.attributeIndex)];
		}
		return current.returnValue;
	}

	public int getInstanceHight(Instance instance) {

		Node current = this.rootNode;
		int hight = 0;  
		while(current.children != null){
			hight++;
			current = current.children[(int) instance.value(current.attributeIndex)];
		}
		return hight;
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

	

	public double calcAverageError(Instances data)
	{
		int wrongClassifications = 0;
		Instance x;
		double classValueOfX = 0.0;

		for (int i = 0; i < data.size(); i++){
			x = data.instance(i);
			classValueOfX = classifyInstance(x);
			if (classValueOfX != x.classValue()){
				wrongClassifications++; // Count the total classification mistakes
			}
		}
		return (((double)wrongClassifications)/data.size());
	}


	//important side effect sets the max hight of tree field 
	public double calcAverageHight(Instances data){

		int maxFound = 0;
		Instance x;
		double hightOfX = 0; //double to prevent future casting 
		double avrgHightOfTree = 0.0;

		for (int i = 0; i < data.size(); i++){

			x = data.instance(i);
			hightOfX = getInstanceHight(x);
			if(hightOfX > maxFound){
				maxFound = (int) hightOfX;
			}
			avrgHightOfTree += hightOfX;
		}
		//set the tree max height field
		this.maxHightOfTree = maxFound;
		return (avrgHightOfTree/data.size());
	}
//---------------------------------------------not orignal code ----------------------------
	/**
     * Print to the console this decision tree structure using 'if statements'
     */
	public void printTree()
    {
    	System.out.println("Root");
    	System.out.println("Returning value: " + rootNode.returnValue);
    	print(rootNode, 1);
    } 
	/**
     * An auxiliary function that prints the given node as an 'if statements' in a recursive process
     * @param node - the node to be printed as an 'if statements'
     * @param level - the node level in the tree(for indentation purposes)
     */
    private void print(Node node, int level)
    {
    	// Check if the given node is a leaf
		if(node.children == null)
		{
			printTabs(level);
			System.out.println("Leaf. Returning value: " + node.returnValue);
			return;
		}
		
		// Loop over the child nodes and print them and their children(recursively)
    	for(int i=0; i<node.children.length; i++)
    	{
    		if(node.children != null && node.children[i] != null)
    		{
    			printTabs(level);
    			System.out.println("if attribute " + node.attributeIndex + " = " + i);
    			
    			if(node.children[i].children != null)
    			{
    				printTabs(level);
    				System.out.println("Returning value: " + node.children[i].returnValue);
    			}
    			
    			print(node.children[i], level + 1);
    		}
    	}
    }
   
    /**
     * print the proper tree indentation
     * @param level - the node level in the tree(for indentation purposes)
     */
    private void printTabs(int level)
    {
    	for(int i=0; i<level; i++)
    	{
    		System.out.print("\t");
    	}
    }
    
//-------------------------------------------end of not orignal code -----------------------
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
	
	//----------------------------------------nivs code ----------------------------------------------
	  /**
     * calculates the impurity measure of the instances in the input using Entropy 
     * @param data
     * @return impurity measure
     */
    private double calcEntropy (Instances data)
	{	
		double entropy = 0;
		int firstClassSize = 0;
		int secondClassSize = 0;
		
		for (int i = 0; i < data.size(); i++)
		{
			if (data.instance(i).classValue() == 0)
				firstClassSize++;
			else secondClassSize++;
		}
		
		if (firstClassSize == 0 | secondClassSize == 0)	return 0;
		
		entropy -= (double)firstClassSize/data.size()*Math.log((double)firstClassSize/data.size());
		entropy -= (double)secondClassSize/data.size()*Math.log((double)secondClassSize/data.size());		
		
		return entropy;
	}
    
    /**
     * calculates the impurity measure of the instances in the input using Gini
     * @param data
     * @return impurity measure
     */
	
	private double calcGini(Instances data)
	{	
		double gini = 1;
		int firstClassSize = 0;
		int secondClassSize = 0;
		
		for (int i = 0; i < data.size(); i++)
		{
			if (data.instance(i).classValue() == 0)
				firstClassSize++;
			else secondClassSize++;
		}
		
		if (firstClassSize == 0 | secondClassSize == 0)	return 0;
		
		gini -= Math.pow((double)firstClassSize / data.size(), 2);
		gini -= Math.pow((double)secondClassSize / data.size(), 2);

		return gini;
	}
	
	
	/**
	 * calculates the gain that will be achieved if we will split the data over the different values of 
	 * the attribute in the index attributeIndex in the given data 
	 * the calculation use the impurity measure that exist in the field impurityMeasure
	 * @param data
	 * @param atributeIndex
	 * @return gain
	 */
	private double calcGain (Instances data, int atributeIndex)
	{
		double gain;
		Instances[] setByAttrNumVal = filterInstancesByAtributteIndex(data, atributeIndex);
		
		if (!this.usingGini)	
			gain = calcEntropy(data);
		else	
			gain = calcGini(data);
		
		for (int i = 0; i < setByAttrNumVal.length; i++)
		{
			if (!this.usingGini)
				gain -= (double)setByAttrNumVal[i].size() / data.size() * calcEntropy(setByAttrNumVal[i]);
			else
				gain -= (double)setByAttrNumVal[i].size() / data.size() * calcGini(setByAttrNumVal[i]);
		}
		
		return gain;
	}
	
	private Instances[] filterInstancesByAtributteIndex(Instances data, int atributteIndex)
	{
		int attributeNumValues = data.attribute(atributteIndex).numValues();
		double[] instancesAttribueteValue = data.attributeToDoubleArray(atributteIndex);
		Instances[] setByAttrNumVal = new Instances[attributeNumValues];
		Instance matchedInstance;
		
		// Initialize instances object 
		for (int i = 0; i < setByAttrNumVal.length; i++) 
		{
			setByAttrNumVal[i] = new Instances(data, data.size());
		}
		
		// Filter the instances to their right position by their attribute value
		for (int i = 0; i < setByAttrNumVal.length; i++) 
		{
			// Loop over all instances and insert the right instance to the right position
			// According to the right attribute value
			for (int j = 0; j < instancesAttribueteValue.length; j++) 
			{
				if(instancesAttribueteValue[j] == i)
				{
					matchedInstance = data.instance(j);
					setByAttrNumVal[i].add(matchedInstance);
				}
			}
		}
		
		return setByAttrNumVal;
	}
	
	/**
	 * find the attribute in the data that provide the highest gain when splitting the 
	 * data according to its different values and return it index
	 * @param data
	 * @return Index of the best attribute
	 */
	private int getBestAttribute(Instances data)
	{
		int numOfAttributes = data.numAttributes() - 1;
		int bestAttributeIndex = 0;
		double curAttributeGain = 0;
		double bestGain = 0;
		
		for (int i = 0; i < numOfAttributes; i++)
		{
			curAttributeGain = calcGain(data, i);
			
			if (curAttributeGain > bestGain)
			{
				bestGain = curAttributeGain;
				bestAttributeIndex = i;
			}
		}
		
		return bestAttributeIndex;
	}
	
}
