package HomeWork1;

import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

public class LinearRegression implements Classifier {

    private int m_ClassIndex;
    private int m_truNumAttributes;
    private double[] m_coefficients;
    private double[] m_firstTeta;
    private double m_alpha;
    private int[] m_best3Atts;
    private double m_err;

    // the method which runs to train the linear regression predictor, i.e.
    // finds its weights.
    /* (non-Javadoc)
     * @see weka.classifiers.Classifier#buildClassifier(weka.core.Instances)
	 */
    @Override
    public void buildClassifier(Instances trainingData) throws Exception {

        // Constructor of the class

        // set the class(target) index
        m_ClassIndex = trainingData.classIndex();

        // set number of the attributes with out the class
        Instance first = trainingData.get(0);
        m_truNumAttributes = first.numAttributes(); //including teta 0

        // initialize teta array with random starting point
        m_coefficients = new double[m_truNumAttributes];
        for (int i = 0; i < m_truNumAttributes; i++) {
            m_coefficients[i] = 1;
        }

        // save starting point
        m_firstTeta = m_coefficients;

        //find the best alpha
        if (trainingData.get(0).numAttributes() > 3) {
            findAlpha(trainingData);
        }

        // run linear regression with correct alpha
        m_coefficients = m_firstTeta;//return to starting point

        m_err = calculateMSE(trainingData);
        double prevError = m_err + 1;
        int counter = 0;

        //loop and check each 100 iterations for stopping condition
        while (Math.abs((prevError - m_err)) > 0.003) {
            m_coefficients = gradientDescent(trainingData);
            counter += 1;
            //update error every 100 iterations
            if (counter % 100 == 0) {
                prevError = m_err;
                m_err = calculateMSE(trainingData);
            }
        }
    }

    private void findAlpha(Instances data) throws Exception {

        // declare alpha array
        double[][] alpha = new double[2][18];
        for (int i = -17; i <= 0; i++) {
            alpha[0][i + 17] = Math.pow(3, i);

        }

        // fill answer array with max number - maybe redundant
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
        return ((gradient * m_alpha) / m);
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
     * @param data
     * @return
     * @throws Exception
     */
    public double calculateMSE(Instances data) throws Exception {

        double mse = 0.0;

        for (int i = 0; i < data.numInstances(); i++) {
            double prediction = regressionPrediction(data.instance(i));
            mse += Math.pow((prediction - data.get(i).value(m_ClassIndex)), 2);
        }
        return (mse / (2 * (data.numInstances())));
    }

    // Generate trio's to identify best attributes subset

    public void findBestTrio(Instances data, double alpha) throws Exception {
        m_best3Atts = new int[3];
        m_alpha = alpha;
        Instances tmpData;
        int tmpTruNumAttributes = data.get(0).numAttributes();
        double err = Integer.MAX_VALUE;
        int[] indexes = new int[4];
        indexes[3] = tmpTruNumAttributes - 1;
        for (int i = 0; i < tmpTruNumAttributes - 1; i++) {
            indexes[0] = i;
            for (int j = i + 1; j < tmpTruNumAttributes - 1; j++) {
                indexes[1] = j;
                for (int k = j + 1; k < tmpTruNumAttributes - 1; k++) {
                    indexes[2] = k;
                    System.out.print("Pass the following attributes: " + data.get(0).attribute(i).toString() + " "
                            + data.get(0).attribute(j).toString() + " " + data.get(0).attribute(k).toString());
                    tmpData = dataSubset(data, indexes);
                    buildClassifier(tmpData);
                    if (calculateMSE(tmpData) < err) {
                        err = calculateMSE(tmpData);
                        m_best3Atts[0] = i;
                        m_best3Atts[1] = j;
                        m_best3Atts[2] = k;
                    }
                    System.out.println(" --> Combination error is: " + calculateMSE(tmpData));
                }
            }
        }
        System.out.println("Training error of the best features " + data.get(0).attribute(m_best3Atts[0]).name() + " "
                + data.get(0).attribute(m_best3Atts[1]).name() + " " + data.get(0).attribute(m_best3Atts[2]).name() + ": " + err);
    }

    private static Instances dataSubset(Instances data, int[] attIndex) throws Exception {
        Instances minData = data;
        // Initialize a Remove object to filter passed dataset to only three attributes
        Remove remove = new Remove();

        // Set @remove to be a subset of @data based on the indexes passed by @attIndex
        remove.setInvertSelection(true);
        remove.setAttributeIndicesArray(attIndex);
        remove.setInputFormat(minData);

        return Filter.useFilter(minData, remove);
    }

    public double getAlpha() {
        return m_alpha;
    }

    public double getError() {
        return m_err;
    }

    public int[] getIndexes() {
        return m_best3Atts;
    }

    public void testError(Instances data, int[] attIndex, double alpha) throws Exception {
        m_alpha = alpha;
        int[] attIndexNew = new int[4];
        for (int i = 0; i < attIndex.length; i++) {
            attIndexNew[i] = attIndex[i];
        }
        attIndexNew[3] = data.get(0).numAttributes() - 1;
        Instances minData = dataSubset(data, attIndexNew);
        buildClassifier(minData);
        System.out.println("Test error of the best features " + data.get(0).attribute(attIndex[0]).name() + " "
                + data.get(0).attribute(attIndex[1]).name() + " " + data.get(0).attribute(attIndex[2]).name() + ": " + m_err);
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
