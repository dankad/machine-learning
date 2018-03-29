package HomeWork1;

import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

public class LinearRegression implements Classifier {

	private int m_ClassIndex;
	private int m_truNumAttributes;
	private double[] m_coefficients;
	private double[] m_firstTeta;
	private double m_alpha;

	// the method which runs to train the linear regression predictor, i.e.
	// finds its weights.
	/* (non-Javadoc)
	 * @see weka.classifiers.Classifier#buildClassifier(weka.core.Instances)
	 */
	@Override
	public void buildClassifier(Instances trainingData) throws Exception {

		// Constructor of the class

		// set the class(target) index
		m_ClassIndex = trainingData.classIndex(); //

		// set number of the attributes with out the class
		Instance first = trainingData.get(0);
		m_truNumAttributes = first.numAttributes(); //including teta 0

		// initialize teta array with random starting point
		m_coefficients = new double[m_truNumAttributes];
		for (int i = 0; i < m_truNumAttributes ; i++) {
			m_coefficients[i] = 1;  //TODO test random setting                            
		}

		// save starting point
		m_firstTeta = m_coefficients;

		//find the best alpha 		
		findAlpha(trainingData);
		
		// run linear regression with correct alpha
		m_coefficients = m_firstTeta;//return to starting point

		double currentError = calculateMSE(trainingData);
		double prevError = currentError + 1;
		int counter  = 0;
		
		//loop and check each 100 iterations for stopping condition
		while (Math.abs((prevError - currentError)) > 0.003) {
			m_coefficients = gradientDescent(trainingData);
			counter += 1;
             //update error every 100 iterations 
			if(counter % 100 == 0){
				prevError = currentError;
				currentError = calculateMSE(trainingData);
			}
		}
// --------------------test printing-------------------------------------------- 
		System.out.println("final alpha " + m_alpha);
		for (int i = 0; i < m_truNumAttributes ; i++) {
			System.out.println("teta num " + i + "is " + m_coefficients[i]);                              
		}
    	System.out.println("final mse: " +  this.calculateMSE(trainingData)); 

	}

	private void findAlpha(Instances data) throws Exception {

		// declare alpha array
		double[][] alpha = new double[2][18];
		for (int i = -17; i <= 0; i++) {
			alpha[0][i + 17] = Math.pow(3, i);

		}

		//fill answer array with max number - maybe redundant    
		for (int i = 0; i <= 17; i++) {
			alpha[1][i] = Integer.MAX_VALUE;
		}

		// main loop run on each alpha
		for (int i = 0; i < alpha[0].length; i++) {
			// Before each alpha reset: starting point, error
			m_coefficients = m_firstTeta;
			m_alpha = alpha[0][i];

			// inner loop does 20000 steps or less
			for (int j = 0; j < 20000; j++) {
				if ((j + 1) % 100 == 0) {
					if (calculateMSE(data) > alpha[1][i] || Math.abs(calculateMSE(data) - alpha[1][i]) < 0.003) {
						break;
					}
					alpha[1][i] = calculateMSE(data);
					m_coefficients = gradientDescent(data);
				}
			}
		}
		// find minimum alpha result
		double min = alpha[1][0];
		int alpha_index = 0;
		for (int k = 1; k < 18; k++) {
			if (min > alpha[1][k]) {
				min = alpha[1][k];
				alpha_index = k;
			}
		}
		m_alpha = alpha[0][alpha_index];
	}

	/**
	 * An implementation of the gradient descent algorithm which should return
	 * the weights of a linear regression predictor which minimizes the average
	 * squared error.
	 * 
	 * @param trainingData
	 * @throws Exception
	 */


	public double[] gradientDescent(Instances trainingData) throws Exception {

		// calculate derivatives for all Tetas
		double[] tempTetaArr = new double[m_coefficients.length];

		for (int i = 0; i < tempTetaArr.length; i++) {
			tempTetaArr[i] = (m_coefficients[i] - calcDerivative(i, trainingData));

		}
		return tempTetaArr; 

	}

	private double calcDerivative(int indexOfTeta, Instances trainingData) throws Exception {

		double gradient = 0.0;
		int m = trainingData.numInstances();

		for (int i = 0; i < m; i++) {
			Instance x = trainingData.get(i);
			if (indexOfTeta == 0) {
				gradient += (regressionPrediction(x) - x.value(m_ClassIndex));
			} else {
				gradient += ((regressionPrediction(x) - x.value(m_ClassIndex)) * x.value(indexOfTeta - 1));
			}
		}
		return ((gradient * m_alpha) / m );
	}

	/**
	 * Returns the prediction of a linear regression predictor with weights
	 * given by m_coefficients on a single instance.
	 *
	 * @param instance
	 * @return
	 * @throws Exception
	 */
	public double regressionPrediction(Instance instance) throws Exception {

		// loop through the teta array and make inner product
		double innerProduct = m_coefficients[0]; // add teta 0

		for (int i = 1; i < m_coefficients.length; i++) {
			innerProduct += (m_coefficients[i] * instance.value(i - 1));
		}
		return innerProduct;
	}

	/**
	 * Calculates the total squared error over the data on a linear regression
	 * predictor with weights given by m_coefficients.
	 *
	 * @param testData
	 * @return
	 * @throws Exception
	 */
	public double calculateMSE(Instances data) throws Exception {

		double mse = 0.0;

		for (int i = 0; i < data.numInstances(); i++) {
			double predection = regressionPrediction(data.instance(i));
			mse += Math.pow((predection - data.get(i).value(m_ClassIndex)), 2);
		}

		return (mse / (  2 * (data.numInstances())));
	}


	@Override
	public double classifyInstance(Instance arg0) throws Exception {
		// Don't change
		return 0;
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
