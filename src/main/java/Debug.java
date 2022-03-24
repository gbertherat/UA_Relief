import weka.classifiers.lazy.IBk;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.Arrays;
import java.util.Comparator;
import java.util.stream.IntStream;

public class Debug{
    public static void main(String[] args) {
        String file = "heart-statlog";
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

                double[] weights = Application.startRelief(dataSetInstances, 10, 5);

                // Récupération des indices des p poids les plus importants
                Integer[] indices = IntStream.range(0, weights.length-1).boxed().toArray(Integer[]::new);
                Arrays.sort(indices, Comparator.<Integer>comparingDouble(i -> weights[i]).reversed());

                int[] indicesToKeep = new int[p+1];
                for(int i = 0; i < p; i++){
                    indicesToKeep[i] = indices[i];
                }
                indicesToKeep[p] = dataSetInstances.numAttributes()-1;
                Arrays.sort(indicesToKeep);

                System.out.println("Fichier: " + file);
                System.out.println("Nombre d'instances: " + dataSetInstances.numInstances());
                System.out.println("Nombre d'itérations de relief: " + (args.length >= 2 ? (Integer.parseInt(args[1]) > dataSetInstances.numInstances() ? dataSetInstances.numInstances() : args[1]) : dataSetInstances.numInstances()));
                System.out.println("Nombre de k plus proche voisins (barycentre): " + (args.length >= 2 ? (Integer.parseInt(args[2]) > dataSetInstances.numInstances() ? dataSetInstances.numInstances() : args[2]) : dataSetInstances.numInstances()));
                System.out.println("Nombre d'attributs avant relief: " + indices.length);
                System.out.println("Nombre d'attributs après relief: " + indicesToKeep.length);
                System.out.println("\n--------\n");

                for(int i = 0; i < dataSetInstances.numAttributes(); i++){
                    System.out.println(i + ") " + dataSetInstances.attribute(i));
                    System.out.println("\tPoids: " + weights[i]);
                }

                System.out.println("\n--------\n");
                System.out.println("Attributs gardées:");
                for(int i = 0; i < dataSetInstances.numAttributes(); i++){
                    int index = i;
                    if(Arrays.stream(indicesToKeep).anyMatch(e -> e == index)) {
                        System.out.println(i + ") " + dataSetInstances.attribute(i).name());
                    }
                }

            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }
}