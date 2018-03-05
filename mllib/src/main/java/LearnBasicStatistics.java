import java.util.ArrayList;
import java.util.Collections;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaDoubleRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.stat.MultivariateStatisticalSummary;
import org.apache.spark.mllib.stat.Statistics;

/**
 * Created by ambarish on 5/3/18.
 */
public class LearnBasicStatistics
{
  public static void main(String[] args)
  {

    /**
     * Summary Statistics
     */
    JavaSparkContext sc = new JavaSparkContext(new SparkConf().setMaster("local[*]").setAppName("LearnBasicStatistics"));

    ArrayList<Vector> vectors = new ArrayList<>();
    vectors.add(Vectors.dense(new double[]{0,2,3,4,5}));
    vectors.add(Vectors.dense(new double[]{2,0,4,5,1}));
    vectors.add(Vectors.dense(new double[]{3,4,0,1,2}));
    vectors.add(Vectors.dense(new double[]{4,5,1,0,3}));
    vectors.add(Vectors.dense(new double[]{5,1,2,3,0}));

    JavaRDD<Vector> mat = sc.parallelize(vectors);

    MultivariateStatisticalSummary summary = Statistics.colStats(mat.rdd());
    System.out.println(summary.mean()); // a dense vector containing the mean value for each column
    System.out.println(summary.variance()); // column-wise variance1
    System.out.println(summary.numNonzeros()); // number of nonzeros in each column

    /**
     * Correlation
     */


    Double[] s1d = new Double[]{1d,2d,3d,4d,5d,6d,7d,8d,9d};
    Double[] s2d = new Double[]{2d,4d,2d,6d,2d,8d,2d,10d,2d};
    ArrayList<Double> doubleSeries1 = new ArrayList<>();
    ArrayList<Double> doubleSeries2 = new ArrayList<>();
    Collections.addAll(doubleSeries1, s1d);
    Collections.addAll(doubleSeries2, s2d);
    JavaDoubleRDD seriesX = sc.parallelizeDoubles(doubleSeries1); // a series
    JavaDoubleRDD seriesY = sc.parallelizeDoubles(doubleSeries2); // a series // must have the same number of
    // partitions and cardinality as seriesX

    // compute the correlation using Pearson's method. Enter "spearman" for Spearman's method. If a
    // method is not specified, Pearson's method will be used by default.
    Double pearson = Statistics.corr(seriesX.srdd(), seriesY.srdd(), "pearson");
    System.out.println("Pearson Correlation of seriesX and seriesY : " + pearson);
    Double spearman = Statistics.corr(seriesX.srdd(), seriesY.srdd(), "spearman");
    System.out.println("Spearman Correlation of seriesX and seriesY : " + spearman);
    JavaRDD<Vector> data = sc.parallelize(vectors); // note that each Vector is a row and not a column

    // calculate the correlation matrix using Pearson's method. Use "spearman" for Spearman's method.
    // If a method is not specified, Pearson's method will be used by default.
    Matrix correlMatrix = Statistics.corr(data.rdd(), "pearson");
    System.out.println("Correl Matrix :\n" +  correlMatrix);


    // Stratified Sampling
    // Hypothesis Testing
    // Random Generator

  }
}
