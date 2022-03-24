import weka.classifiers.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.Remove;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Random;
import java.util.stream.IntStream;

public class Debug{
    public static void main(String[] args) {
        String file = "optdigits_39";
        InputStream dataSetStream = Application.class.getClassLoader().getResourceAsStream(file + ".arff");

        if(dataSetStream != null) {
            BufferedReader dataSet = new BufferedReader(new InputStreamReader(dataSetStream));
            IBk kNNClassifier = new IBk();
            kNNClassifier.setKNN(5);

            try {
                Instances dataSetInstances = new Instances(dataSet);
                dataSetInstances.setClassIndex(dataSetInstances.numAttributes() - 1);

                Normalize normalize = new Normalize();
                normalize.setInputFormat(dataSetInstances);
                dataSetInstances = Filter.useFilter(dataSetInstances, normalize);

                int p = (int) ((2/(float) 3) * dataSetInstances.numAttributes());

                double[] weights;
                if(args.length == 2) {
                    int m = Integer.parseInt(args[1]);
                    weights = Application.startRelief(dataSetInstances, m);
                } else {
                    weights = Application.startRelief(dataSetInstances);
                }

                // Récupération des indices des p poids les plus importants
                Integer[] indices = IntStream.range(0, weights.length-1).boxed().toArray(Integer[]::new);
                Arrays.sort(indices, Comparator.<Integer>comparingDouble(i -> weights[i]).reversed());

                int[] indicesToKeep = new int[p];
                for(int i = 0; i < p-1; i++){
                    indicesToKeep[i] = indices[i];
                }
                indicesToKeep[p-1] = weights.length-1;
                Arrays.sort(indicesToKeep);

                System.out.println("Fichier: " + file);
                System.out.println("Nombre d'instances: " + dataSetInstances.numInstances());
                System.out.println("Nombre d'itérations de relief: " + (args.length == 2 ? (Integer.parseInt(args[1]) > dataSetInstances.numInstances() ? dataSetInstances.numInstances() : args[1]) : dataSetInstances.numInstances()));
                System.out.println("Nombre d'attributs avant relief: " + indices.length);
                System.out.println("Nombre d'attributs après relief: " + indicesToKeep.length);
                System.out.println("Classifier kNN avec k = " + kNNClassifier.getKNN());
                System.out.println("\n--------");

                // Evaluation avant le filtre
                Evaluation evaluationBeforeFilter = new Evaluation(dataSetInstances);
                evaluationBeforeFilter.crossValidateModel(kNNClassifier, dataSetInstances, 10, new Random(1));
                System.out.println(evaluationBeforeFilter.toSummaryString("\nResults before filter:\n", false));
                System.out.println("--------");

                // Création du filtre
                Remove removeFilter = new Remove();
                removeFilter.setAttributeIndicesArray(indicesToKeep);
                removeFilter.setInvertSelection(true);
                removeFilter.setInputFormat(dataSetInstances);

                // Application du filtre
                Instances data = Filter.useFilter(dataSetInstances, removeFilter);
                data.setClassIndex(data.numAttributes()-1);

                // Evaluation après filtre
                Evaluation evaluationAfterFilter = new Evaluation(data);
                evaluationAfterFilter.crossValidateModel(kNNClassifier, data, 10, new Random(1));

                System.out.println(evaluationAfterFilter.toSummaryString("\nResults after filter:\n", false));

            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }
}