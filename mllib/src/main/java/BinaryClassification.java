import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.classification.SVMModel;
import org.apache.spark.mllib.classification.SVMWithSGD;
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.GeneralizedLinearModel;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.regression.LinearRegressionModel;
import org.apache.spark.mllib.regression.LinearRegressionWithSGD;

import scala.Tuple2;

/**
 * Binary Classification Algorithms
 * 1) Linear SVM
 * 2) Linear Regression
 */
public class BinaryClassification
{
  public static void main(String[] args)
  {

    // Setup Spark Context
    SparkConf sparkConf = new SparkConf().setMaster("local[*]").setAppName("BinaryClassification");
    JavaSparkContext sc = new JavaSparkContext(sparkConf);

    // Read data into RDD's
    String trainingData = BinaryClassification.class.getClassLoader().getResource("data/binary.txt").getFile();
    JavaRDD<String> data = sc.textFile(trainingData);
    JavaRDD<LabeledPoint> parsedData = data.map(
      (Function<String, LabeledPoint>)line -> {
        String[] columns = line.split(",");
        String label = columns[0];
        double[] v = new double[columns.length-1];
        for (int i = 1; i < columns.length - 1; i++)
          v[i] = Double.parseDouble(columns[i]);
        return new LabeledPoint(Double.parseDouble(label), Vectors.dense(v));
      }
    );

    // Print Read Data
    parsedData.take(10).forEach(lp -> {
      System.out.println(lp.toString());
    });

    // Split Data into training and testing

    JavaRDD<LabeledPoint> dataTrain = parsedData.sample(false, 0.6);
    JavaRDD<LabeledPoint> dataTest = parsedData.subtract(dataTrain);

    /**
     * Linear SVM Model
     */

    // Initialize Model
    int numIterations = 100;
    SVMModel model = SVMWithSGD.train(dataTrain.rdd(), numIterations);
    // Save model to file
    model.save(sc.sc(), "svm_model");
    // Export to PMML
    model.toPMML("svm_model.pmml");
    model.clearThreshold();

    // Compute  raw scores on the test set.
    JavaRDD<Tuple2<Object, Object>> scoreAndLabels = getScoreAndLabels(dataTest, model);

    // Get evaluation metrics.
    BinaryClassificationMetrics metrics =
      new BinaryClassificationMetrics(JavaRDD.toRDD(scoreAndLabels));
    double auROC = metrics.areaUnderROC();
    System.out.println("Area under ROC for LSVM = " + auROC);

    /**
     * Linear Regression Model
     */

    LinearRegressionModel lrModel = LinearRegressionWithSGD.train(dataTrain.rdd(), numIterations);
    lrModel.save(sc.sc(), "lr_model");
    lrModel.toPMML("lr_model.pmml");

    // Compute  raw scores on the test set.
    JavaRDD<Tuple2<Object, Object>> scoreAndLabelsLr = getScoreAndLabels(dataTest, lrModel);

    BinaryClassificationMetrics metricsLr =
      new BinaryClassificationMetrics(JavaRDD.toRDD(scoreAndLabelsLr));
    double auROCLr = metrics.areaUnderROC();
    System.out.println("Area under ROC for LR = " + auROCLr);

  }

  // Util Method
  static JavaRDD<Tuple2<Object, Object>> getScoreAndLabels(JavaRDD<LabeledPoint> testData, GeneralizedLinearModel  model){
    JavaRDD<Tuple2<Object, Object>> scoreAndLabels = testData.map(
      (Function<LabeledPoint, Tuple2<Object, Object>>)p -> {
        Double score = model.predict(p.features());
        return new Tuple2<Object, Object>(score, p.label());
      }
    );
    return  scoreAndLabels;
  }

}
