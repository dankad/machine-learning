package HomeWork3;

import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

class DistanceCalculator {
    /**
     * We leave it up to you whether you want the distance method to get all relevant
     * parameters(lp, efficient, etc..) or have it as a class variables.
     */

    private int m_PValue;

    public double distance(Instance one, Instance two) {
        return 0.0;
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

    public enum DistanceCheck {Regular, Efficient}

    public int m_k;

    private Instances m_trainingInstances;

    @Override
    /**
     * Build the knn classifier. In our case, simply stores the given instances for 
     * later use in the prediction.
     * @param instances
     */
    public void buildClassifier(Instances instances) throws Exception {
    }

    /**
     * Returns the knn prediction on the given instance.
     *
     * @param instance
     * @return The instance predicted value.
     */
    public double regressionPrediction(Instance instance) {
        return 0.0;
    }

    /**
     * Caclcualtes the average error on a give set of instances.
     * The average error is the average absolute error between the target value and the predicted
     * value across all insatnces.
     *
     * @param insatnces
     * @return
     */
    public double calcAvgError(Instances insatnces) {
        return 0.0;
    }

    /**
     * Calculates the cross validation error, the average error on all folds.
     *
     * @param insances     Insances used for the cross validation
     * @param num_of_folds The number of folds to use.
     * @return The cross validation error.
     */
    public double crossValidationError(Instances insances, int num_of_folds) {
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
            tempDist = lpDistance(instance, m_trainingInstances.instance(i));
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
    public double getAverageValue(/* Collection of your choice */) {
        return 0.0;
    }

    /**
     * Calculates the weighted average of the target values of all the elements in the collection
     * with respect to their distance from a specific instance.
     *
     * @return
     */
    public double getWeightedAverageValue(/* Collection of your choice */) {
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
