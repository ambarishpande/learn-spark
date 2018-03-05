import java.util.ArrayList;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.linalg.DenseMatrix;
import org.apache.spark.ml.linalg.Matrices;
import org.apache.spark.ml.linalg.Matrix;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.linalg.distributed.IndexedRow;
import org.apache.spark.mllib.linalg.distributed.IndexedRowMatrix;
import org.apache.spark.mllib.linalg.distributed.RowMatrix;
import org.apache.spark.mllib.regression.LabeledPoint;

/**
 * Learn Spark ML Datatypes.
 */

public class LearnDataTypes
{
  public static void main(String[] args) {

    /**
     *  Vector Datatype
     *  A local vector has integer-typed and 0-based indices and double-typed values, stored on a single machine.
     */
    System.out.println("+++++++++++++++++++++++++++++++++");
    System.out.println("LOCAL VECTOR DATATYPE IN SPARK ML");
    System.out.println("+++++++++++++++++++++++++++++++++");
    // Create a dense vector (1.0, 3.0).
    Vector dv = Vectors.dense(1.0,  3.0);
    System.out.println("Dense Vector 1: " + dv);
    // Create a dense vector (2.0, 6.0).
    Vector dv2 = Vectors.dense(2.0,  6.0);
    System.out.println("Dense Vector 2: " + dv2);
    // Square Cartesian distance between two vectors
    System.out.println("Cartesian distance dv1^2 + dv2^2 = " + Vectors.sqdist(dv,dv2));
    // Create a sparse vector (1.0, 0.0, 3.0) by specifying its indices and values corresponding to nonzero entries.
    Vector sv = Vectors.sparse(3, new int[] {0, 2}, new double[] {1.0, 3.0});
    System.out.println("Sparse Vector " + sv);
    // Zero Vector
    Vector zero = Vectors.zeros(10);
    System.out.println("Zero Vector " + zero);
    // Vector from String.
    Vector fromString = Vectors.parse("[5,4,3,2,1]");
    System.out.println("Vector from String "  + fromString);
    //Vector from double array
    double[] doubleArray = new double[]{4.3, 5.6, 6.8, 8.9};
    Vector fromDouble = Vectors.dense(doubleArray);
    System.out.println("Vector from Double array "  + fromDouble);

    /**
     * Labelled Point
     * A labeled point is a local vector, either dense or sparse, associated with a label/response.
     */

    System.out.println("+++++++++++++++++++++++++++++++++++");
    System.out.println("LABELED POINT DATATYPE IN SPARK ML");
    System.out.println("+++++++++++++++++++++++++++++++++++");

    // Create a labeled point with a positive label and a dense feature vector.
    LabeledPoint pos = new LabeledPoint(1.0, Vectors.dense(1.0, 0.0, 3.0));
    System.out.println("Positive Labelled Point " + pos);
    System.out.println("Features : " + pos.features() + "  Label : " + pos.label());
    // Create a labeled point with a negative label and a sparse feature vector.
    LabeledPoint neg = new LabeledPoint(0.0, Vectors.sparse(3, new int[] {0, 2}, new double[] {1.0, 3.0}));
    System.out.println("Negative Labelled Point " + neg);


    /**
     * Local Matrix
     * A local matrix has integer-typed row and column indices and double-typed values, stored on a single machine.
     */
    System.out.println("+++++++++++++++++++++++++++++++++++");
    System.out.println("LOCAL MATRIX DATATYPE IN SPARK ML");
    System.out.println("+++++++++++++++++++++++++++++++++++");

    // Stored in Column Major Format.
    Matrix dm = Matrices.dense(3,3,new double[]{1,2,3,4,5,6,7,8,9});
    System.out.println("3x3 Dense Matrix 1\n" + dm);
    // Transpose of matrix
    System.out.println("Transpose\n" + dm.transpose());
    // Multiplication Of Matrices
    Matrix dm2 = Matrices.dense(3,3, new double[]{1,4,7,2,5,8,3,6,9});
    System.out.println("3x3 Dense Matrix 2\n" + dm);
    System.out.println("Multiplication dm1 X dm2\n" + dm.multiply((DenseMatrix) dm2));

    /**
     * Distributed Matrix
     * A distributed matrix has long-typed row and column indices and double-typed values, stored distributively in one or more RDDs.
     */

    System.out.println("++++++++++++++++++++++++++++++++++++++++");
    System.out.println("DISTRIBUTED MATRIX DATATYPE IN SPARK ML");
    System.out.println("++++++++++++++++++++++++++++++++++++++++");


    JavaSparkContext sc = new JavaSparkContext(new SparkConf().setMaster("local[*]").setAppName("DataTypeApp"));

    // Row Matrix
    ArrayList<Vector> vectors = new ArrayList<>();
    vectors.add(dv);
    vectors.add(dv2);
    JavaRDD<Vector> rows = sc.parallelize(vectors);
    RowMatrix mat = new RowMatrix(rows.rdd());
    System.out.println("Mean : " + mat.computeColumnSummaryStatistics().mean());
    System.out.println("Max : " + mat.computeColumnSummaryStatistics().max());
    System.out.println("Variance : " + mat.computeColumnSummaryStatistics().variance());
    System.out.println("Min : " + mat.computeColumnSummaryStatistics().min());
    System.out.println("Count : " + mat.computeColumnSummaryStatistics().count());

    // IndexedRowMatrix
    ArrayList<IndexedRow> indexedVectors = new ArrayList<>();
    indexedVectors.add(new IndexedRow(0, dv));
    indexedVectors.add(new IndexedRow(1, dv2));
    JavaRDD<IndexedRow> indexedRows =  sc.parallelize(indexedVectors);
    IndexedRowMatrix indexedRowMatrix = new IndexedRowMatrix(indexedRows.rdd());

    // Coordinate / Block Matrix

    // Fin.

  }
}