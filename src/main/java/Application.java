import weka.classifiers.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.core.*;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.Remove;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Random;
import java.util.stream.IntStream;

public class Application {
    public static Instance getNearestInstance(Instances classes, Instance instance){
        double minDist = Double.MAX_VALUE;
        Instance nearest = null;

        NormalizableDistance distance = new EuclideanDistance(classes);
        for(Instance inst : classes) {
            double dist = distance.distance(instance, inst);
            if(dist < minDist){
                minDist = dist;
                nearest = inst;
            }
        }

        return nearest;
    }

    public static Instance getBarycentre(Instances classes, Instance instance, int k){
        if(k > classes.numInstances()){
            k = classes.numInstances();
        }

        ArrayList<Instance> nearestArr = new ArrayList<>();
        for(int j = 0; j < k; j++) {
            Instance nearestHit = getNearestInstance(classes, instance);
            nearestArr.add(nearestHit);
            classes.remove(nearestHit);
        }

        Instance barycentre = (Instance) instance.copy();
        for(int j = 0; j < barycentre.numAttributes(); j++){
            double valAttribute = 0;
            for(Instance inst: nearestArr){
                valAttribute += inst.value(j);
            }
            double mean = valAttribute / nearestArr.size();
            barycentre.setValue(j, mean);
        }

        return barycentre;
    }



    public static double[] startRelief(Instances instances){
        return relief(instances, instances.numInstances(), 5, false);
    }

    public  static double[] startRelief(Instances instances, int m){
        return relief(instances, m, 5,m > instances.numInstances());
    }

    public static double[] startRelief(Instances instances, int m, int k){
        return relief(instances, m, k,m > instances.numInstances());
    }

    public static double[] relief(Instances instances, int m, int k, boolean useRandom){
        int d = instances.numAttributes();

        double[] weights = new double[d];
        for(int i = 0; i < d; i++){
            weights[i] = 0;
        }

        for(int i = 0; i < m; i++) {
            Instance randInstance;
            if (useRandom) {
                randInstance = instances.instance(new Random().nextInt(instances.numInstances()));
            } else {
                randInstance = instances.instance(i);
            }

            Instances positiveClasses = new Instances(instances);
            positiveClasses.remove(instances.indexOf(randInstance));
            positiveClasses.removeIf(e -> e.classValue() != randInstance.classValue());

            Instances negativeClasses = new Instances(instances);
            negativeClasses.remove(instances.indexOf(randInstance));
            negativeClasses.removeIf(e -> e.classValue() == randInstance.classValue());

            Instance hitBarycentre;
            Instance missBarycentre;
            if(k > 1) {
                hitBarycentre = getBarycentre(positiveClasses, randInstance, k);
                missBarycentre = getBarycentre(negativeClasses, randInstance, k);
            } else {
                hitBarycentre = getNearestInstance(positiveClasses, randInstance);
                missBarycentre = getNearestInstance(negativeClasses, randInstance);
            }

            for(int j = 0; j < weights.length; j++){
                weights[j] = weights[j] - (1/(float) m) * Math.pow((randInstance.value(j) - hitBarycentre.value(j)), 2)
                                        + (1/(float) m) * Math.pow((randInstance.value(j) - missBarycentre.value(j)), 2);
            }

        }

        return weights;
    }

    private static void error(int type){
        System.out.println("Argument(s) invalide(s)!");
        if(type == 0) {
            System.out.println("Nom de fichiers invalide!");
            System.out.println("Fichiers disponibles:\n heart-statlog\n iris2Classes \n optdigits_39");
        } else if(type == 1){
            System.out.println("Nombre m invalide!");
            System.out.println("Veuillez choisir une valeur supérieure à 0");
        } else if(type == 2) {
            System.out.println("Nombre k invalide!");
            System.out.println("Veuillez choisir une valeur supérieure à 0");
        } else if(type == 3){
            System.out.println("Vous avez n'avez pas assez d'arguments!");
        } else {
            System.out.println("Vous avez trop d'arguments!");
        }
        System.exit(-type);
    }

    public static void main(String[] args) {
        if(args.length < 1 || args.length > 3){
            if(args.length < 1){
                error(3);
            } else {
                error(4);
            }
        }

        String file = args[0];
        if(!file.equals("heart-statlog") && !file.equals("iris2Classes") && !file.equals("optdigits_39")){
            error(0);
        }

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
                if(args.length == 3){
                    int m = Integer.parseInt(args[1]);
                    if(m < 1){
                        error(1);
                    }
                    int k = Integer.parseInt(args[2]);
                    if(k < 1){
                        error(2);
                    }
                    weights = startRelief(dataSetInstances, m, k);
                } else if(args.length == 2) {
                    int m = Integer.parseInt(args[1]);
                    weights = startRelief(dataSetInstances, m);
                } else {
                    weights = startRelief(dataSetInstances);
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
