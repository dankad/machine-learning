package HomeWork3;

import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.supervised.instance.*;

class DistanceCalculator {

    private boolean m_Efficient;
    private boolean m_Infinity;
    private int m_PValue;
    public double m_threshold = Double.MAX_VALUE;

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
            result = efficientLpDistance(one, two);
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
        for (int i = 0; i < one.numAttributes() - 1; i++) {
            sumLp += Math.pow(Math.abs(oneValues[i] - twoValues[i]), m_PValue);
        }
        return Math.pow(sumLp, 1 / (double) m_PValue);
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
        for (int i = 0; i < one.numAttributes() - 1; i++) {
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
        double[] oneValues = one.toDoubleArray();
        double[] twoValues = two.toDoubleArray();

        double sumLp = 0;
        for (int i = 0; i < one.numAttributes() - 1; i++) {
            sumLp += Math.pow(Math.abs(oneValues[i] - twoValues[i]), m_PValue);
            if (sumLp > m_threshold) {
                sumLp = Double.MAX_VALUE;
                break;
            }
        }
        return Math.pow(sumLp, 1 / (double) m_PValue);
    }

    /**
     * Returns the Lp distance between 2 instances, while using an efficient distance check.
     *
     * @param one
     * @param two
     * @return
     */
    private double efficientLInfinityDistance(Instance one, Instance two) {
        double[] oneValues = one.toDoubleArray();
        double[] twoValues = two.toDoubleArray();
        double tempDiff, max = Integer.MIN_VALUE;
        for (int i = 0; i < one.numAttributes() - 1; i++) {
            tempDiff = Math.abs(oneValues[i] - twoValues[i]);
            max = tempDiff > max ? tempDiff : max;
            if (max > m_threshold) {
                return Double.MAX_VALUE;
            }
        }
        return max;
    }
}

public class Knn implements Classifier {


    //public enum DistanceCheck{Regular, Efficient}

    public Instances m_trainingInstances;
    private boolean useWeight;

    //all distances related fields
    private DistanceCalculator m_DistanceCalculator;
    private int m_PValue;
    private boolean m_InfinityP;
    public boolean m_Efficient;
    private int m_k;

    public Knn(boolean weight, int PValue, boolean infinity, int k) {
        useWeight = weight;
        m_PValue = PValue;
        m_InfinityP = infinity;
        m_k = k;
    }

    @Override
    /**
     * Build the knn classifier. In our case, simply stores the given instances for 
     * later use in the prediction.
     * @param instances
     */
    public void buildClassifier(Instances instances) throws Exception {
        //need to check if more fields are needed - p distancecheck ect
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
        // System.out.println("k index is: " + kNearestIndex[0]);
        double answer;

        //use average class or weighted method to calculate class
        answer = !useWeight ? getAverageValue(kNearestIndex) : getWeightedAverageValue(kNearestIndex, instance);

        return answer;
    }

    /**
     * Calculates the average error on a given set of instances.
     * The average error is the average absolute error between the target value and the predicted
     * value across all instances.
     *
     * @param instances
     * @return
     */
    public double calcAvgError(Instances instances) {
        double calcErrors = 0;
        for (int i = 0; i < instances.size(); i++) {
            calcErrors += Math.abs(instances.get(i).classValue() - regressionPrediction(instances.get(i)));
        }
        return calcErrors / instances.size();
    }

    /**
     * Calculates the cross validation error, the average error on all folds.
     *
     * @param instances    Instances used for the cross validation
     * @param num_of_folds The number of folds to use.
     * @return The cross validation error.
     */
    public double crossValidationError(Instances instances, int num_of_folds) throws Exception {
        double sumErrors = 0.0;
        Instances trainingData;
        Instances testData;
        StratifiedRemoveFolds folds = new StratifiedRemoveFolds();
        folds.setNumFolds(num_of_folds);

        for (int i = 1; i <= num_of_folds; i++) {
            //remove i'th fold from dataset
            folds.setFold(i);
            folds.setInputFormat(instances);
            folds.setInvertSelection(true);
            trainingData = Filter.useFilter(instances, folds);

            //create new testing data for cross validation
            folds.setInputFormat(instances);
            folds.setInvertSelection(false);
            testData = Filter.useFilter(instances, folds);
            buildClassifier(trainingData);
            sumErrors += calcAvgError(testData);
        }
        return sumErrors / num_of_folds;
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

        //for loop insert max size for all instances - relevant for efficient
        for (int i = 0; i < m_trainingInstances.size(); i++) {
            distForNeighbors[i][1] = Double.MAX_VALUE;
        }

        // calculate the distance of @instance from all instances in the data set
        for (int i = 0; i < m_trainingInstances.size(); i++) {
            tempDist = m_DistanceCalculator.distance(instance, m_trainingInstances.instance(i));

            //if the efficient flag is up after every calculation up date threshold
            if (m_Efficient) {
                int n = m_trainingInstances.size();
                for (int m = 1; m < n; m++) {
                    double[] tempNeighbor = distForNeighbors[m];
                    int j = m - 1;
                    while (j >= 0 && distForNeighbors[j][1] > tempNeighbor[1]) {
                        distForNeighbors[j + 1] = distForNeighbors[j];
                        j = j - 1;
                    }
                    distForNeighbors[j + 1] = tempNeighbor;
                }
                int indexOfTheKfar = (int) distForNeighbors[m_k][0];
                m_DistanceCalculator.m_threshold = m_DistanceCalculator.distance(instance, m_trainingInstances.get(indexOfTheKfar));
            }

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
     * Calculates the average value of the given elements in the collection.
     *
     * @param
     * @return
     */
    public double getAverageValue(int[] kNearest) {
        double sum = 0.0;
        //sum all class value of all knn's
        for (int i = 0; i < kNearest.length; i++) {
            sum += this.m_trainingInstances.get(kNearest[i]).classValue();
        }
        return sum / kNearest.length;
    }

    /**
     * Calculates the weighted average of the target values of all the elements in the collection
     * with respect to their distance from a specific instance.
     *
     * @return
     */
    public double getWeightedAverageValue(int[] kNearest, Instance instance) {
        double tmpDist, sum = 0.0;
        double sumOfSquares = 0.0;
        //sum all class value of all knn's
        for (int i = 0; i < kNearest.length; i++) {
            tmpDist = m_DistanceCalculator.distance(instance, m_trainingInstances.instance(kNearest[i]));
            if (tmpDist == 0) return this.m_trainingInstances.get(kNearest[i]).classValue();
            // should return exactly the value of the compared instance if distance is zero
            sum += this.m_trainingInstances.get(kNearest[i]).classValue() / Math.pow(tmpDist, 2);
            sumOfSquares += 1 / Math.pow(tmpDist, 2);
        }
        return sum / sumOfSquares;
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
