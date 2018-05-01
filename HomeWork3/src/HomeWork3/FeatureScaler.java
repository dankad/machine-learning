package HomeWork3;

import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Standardize;

public class FeatureScaler {
    /**
     * Returns a scaled version (using standardized normalization) of the given dataset.
     *
     * @param instances The original dataset.
     * @return A scaled instances object.
     */
    public Instances scaleData(Instances instances) throws Exception {
        Standardize standardizedAttributes = new Standardize();
        standardizedAttributes.setInputFormat(instances);
        Instances scaledInstances = Filter.useFilter(instances, standardizedAttributes);

        return scaledInstances;
    }
}