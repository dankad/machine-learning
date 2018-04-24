package HomeWork3;

import weka.core.Instance;
import weka.core.Instances;

public class FeatureScaler {
    /**
     * Returns a scaled version (using standardized normalization) of the given dataset.
     *
     * @param instances The original dataset.
     * @return A scaled instances object.
     */
    public Instances scaleData(Instances instances) {
        Instances scaledInstances = instances;
        int numOfInstances = scaledInstances.numInstances();
        int numOfAttributes = scaledInstances.numAttributes();
        double std_x, mean_x, scaledValue_x;
        Instance currInstance;
        for (int i = 0; i < numOfAttributes; i++) {
            std_x = Math.pow(scaledInstances.variance(i), 2);
            mean_x = scaledInstances.meanOrMode(i);
            for (int j = 0; j < numOfInstances; j++) {
                currInstance = scaledInstances.instance(j);
                scaledValue_x = (currInstance.value(i) - mean_x) / std_x;
                currInstance.setValue(i, scaledValue_x);
            }
        }
        return scaledInstances;
    }
}