package HomeWork3;

import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

class DistanceCalculator {

    private boolean m_Efficient;
    private boolean m_Infinity;
    private int m_PValue;

    // constructor for the distance calculator object
    public DistanceCalculator(int p, boolean efficient, boolean infinity) {
        m_Efficient = efficient;
        m_Infinity = infinity;
        m_PValue = p;
    }

    /**
     * We leave it up to you whether you want the distance method to get all relevant
     * parameters(lp, efficient, etc..) or have it as a class variables.
     */

    public double distance(Instance one, Instance two) {
        double result = 0;
        if (!m_Infinity && !m_Efficient) {
            result = lpDistance(one, two);
        } else if (m_Infinity && !m_Efficient) {
            result = lInfinityDistance(one, two);
        } else if (!m_Infinity && m_Efficient) {
            result = efficientLInfinityDistance(one, two);
        } else if (m_Infinity && m_Efficient) {
            result = efficientLInfinityDistance(one, two);
        }
        return result;
    }

    /**
     * Returns the Lp distance between 2 instances.
     *
     * @param one
     * @param two
     */
    private double lpDistance(Instance one, Instance two) {
        double[] oneValues = one.toDoubleArray();
        double[] twoValues = two.toDoubleArray();
        double sumLp = 0;
        for (int i = 0; i < one.numAttributes(); i++) {
            sumLp += Math.pow((oneValues[i] - twoValues[i]), m_PValue);
        }
        return Math.pow(sumLp, 1 / m_PValue);
    }

    /**
     * Returns the L infinity distance between 2 instances.
     *
     * @param one
     * @param two
     * @return
     */
    private double lInfinityDistance(Instance one, Instance two) {
        double[] oneValues = one.toDoubleArray();
        double[] twoValues = two.toDoubleArray();
        double tempDiff, max = Integer.MIN_VALUE;
        for (int i = 0; i < one.numAttributes(); i++) {
            tempDiff = Math.abs(oneValues[i] - twoValues[i]);
            max = tempDiff > max ? tempDiff : max;
        }
        return max;
    }

    /**
     * Returns the Lp distance between 2 instances, while using an efficient distance check.
     *
     * @param one
     * @param two
     * @return
     */
    private double efficientLpDistance(Instance one, Instance two) {
        return 0.0;
    }

    /**
     * Returns the Lp distance between 2 instances, while using an efficient distance check.
     *
     * @param one
     * @param two
     * @return
     */
    private double efficientLInfinityDistance(Instance one, Instance two) {
        return 0.0;
    }
}

public class Knn implements Classifier {


    //public enum DistanceCheck{Regular, Efficient}

    private Instances m_trainingInstances;
    private boolean useWeight;

    //all distances related fields
    private DistanceCalculator m_DistanceCalculator;
    private int m_PValue;
    private boolean m_InfinityP;
    private boolean m_Efficient;
    private int m_k;

    @Override
    /**
     * Build the knn classifier. In our case, simply stores the given instances for 
     * later use in the prediction.
     * @param instances
     */
    public void buildClassifier(Instances instances) throws Exception {
        //need to check if more fields are needed - p distancecheck ect
        //we might be missing something
        this.m_trainingInstances = instances;
        this.m_DistanceCalculator = new DistanceCalculator(m_PValue, m_InfinityP, m_Efficient);
    }

    /**
     * Returns the knn prediction on the given instance.
     *
     * @param instance
     * @return The instance predicted value.
     */
    public double regressionPrediction(Instance instance) {
        int[] kNearestIndex = findNearestNeighbors(instance);
        double answer;

        //use average class or weighted method to calculate class
        answer = !useWeight ? getAverageValue(kNearestIndex) : getWeightedAverageValue(kNearestIndex);

        return answer;
    }

    /**
     * Caclcualtes the average error on a give set of instances.
     * The average error is the average absolute error between the target value and the predicted
     * value across all insatnces.
     *
     * @param instances
     * @return
     */
    public double calcAvgError(Instances instances) {
        return 0.0;
    }

    /**
     * Calculates the cross validation error, the average error on all folds.
     *
     * @param instances    Insances used for the cross validation
     * @param num_of_folds The number of folds to use.
     * @return The cross validation error.
     */
    public double crossValidationError(Instances instances, int num_of_folds) {
        return 0.0;
    }


    /**
     * Finds the k nearest neighbors.
     *
     * @param instance
     */
    public int[] findNearestNeighbors(Instance instance) {
        double[][] distForNeighbors = new double[m_trainingInstances.size()][2];
        double tempDist;
        int[] nearestNeighbors = new int[m_k];

        // calculate the distance of @instance from all instance in the data set
        for (int i = 0; i < m_trainingInstances.size(); i++) {
            tempDist = m_DistanceCalculator.distance(instance, m_trainingInstances.instance(i));
            if (tempDist != 0) {
                distForNeighbors[i][0] = i;
                distForNeighbors[i][1] = tempDist;
            } else {
                distForNeighbors[i][0] = i;
                distForNeighbors[i][1] = Integer.MAX_VALUE;
            }
        }

        // sort indexes by dist from @instance
        int n = m_trainingInstances.size();
        for (int i = 0; i < n; i++) {
            double[] tempNeighbor = distForNeighbors[i];
            int j = i - 1;
            while (j >= 0 && distForNeighbors[j][1] > tempNeighbor[1]) {
                distForNeighbors[j + 1] = distForNeighbors[j];
                j = j - 1;
            }
            distForNeighbors[j + 1] = tempNeighbor;
        }

        // set the array with the indexes of nearest neighbors
        for (int i = 0; i < m_k; i++) {
            nearestNeighbors[i] = (int) distForNeighbors[i][0];
        }
        return nearestNeighbors;
    }

    /**
     * Cacluates the average value of the given elements in the collection.
     *
     * @param
     * @return
     */
    public double getAverageValue(int[] kNearest) {
        return 0.0;
    }

    /**
     * Calculates the weighted average of the target values of all the elements in the collection
     * with respect to their distance from a specific instance.
     *
     * @return
     */
    public double getWeightedAverageValue(int[] kNearest) {
        return 0.0;
    }


    @Override
    public double[] distributionForInstance(Instance arg0) throws Exception {
        // TODO Auto-generated method stub - You can ignore.
        return null;
    }

    @Override
    public Capabilities getCapabilities() {
        // TODO Auto-generated method stub - You can ignore.
        return null;
    }

    @Override
    public double classifyInstance(Instance instance) {
        // TODO Auto-generated method stub - You can ignore.
        return 0.0;
    }
}
