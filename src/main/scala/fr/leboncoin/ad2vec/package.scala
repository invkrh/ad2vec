package fr.leboncoin

import com.github.fommil.netlib.F2jBLAS
import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vector}
import org.apache.spark.sql.SparkSession

package object ad2vec {

  def init(): Unit = {}

  //////////////////////////////////////////////////////////////////////////////
  // Spark
  //////////////////////////////////////////////////////////////////////////////

  val spark = SparkSession.builder().appName("ad2vec").master("local[*]").getOrCreate()

  //  val sc = spark.sparkContext

  /**
   * Efficient parquet committer
   */
  //  sc.hadoopConfiguration.set(
  //    "spark.sql.parquet.output.committer.class",
  //    "org.apache.spark.sql.parquet.DirectParquetOutputCommitter")

  /**
   * AWS credential
   */
  //  val chain =
  //    new AWSCredentialsProviderChain(
  //      new EnvironmentVariableCredentialsProvider,
  //      new SystemPropertiesCredentialsProvider)
  //  sc.hadoopConfiguration.set("fs.s3a.access.key", chain.getCredentials.getAWSAccessKeyId)
  //  sc.hadoopConfiguration.set("fs.s3a.secret.key", chain.getCredentials.getAWSSecretKey)

  val sqlContext = spark.sqlContext

  //////////////////////////////////////////////////////////////////////////////
  // Algebra
  //////////////////////////////////////////////////////////////////////////////

  // Use reflection to use spark code

  def nonNegativeMod(x: Int, mod: Int): Int = {
    val rawMod = x % mod
    rawMod + (if (rawMod < 0) mod else 0)
  }

  val f2jBLAS = new F2jBLAS

  /**
   * y += a * x
   */
  def axpy(a: Double, x: Vector, y: Vector): Unit = {
    require(x.size == y.size)
    y match {
      case dy: DenseVector =>
        x match {
          case sx: SparseVector =>
            axpy(a, sx, dy)
          case dx: DenseVector =>
            axpy(a, dx, dy)
          case _ =>
            throw new UnsupportedOperationException(s"axpy doesn't support x type ${x.getClass}.")
        }
      case _ =>
        throw new IllegalArgumentException(
          s"axpy only supports adding to a dense vector but got type ${y.getClass}.")
    }
  }

  private def axpy(a: Double, x: SparseVector, y: DenseVector): Unit = {
    val xValues = x.values
    val xIndices = x.indices
    val yValues = y.values
    val nnz = xIndices.length

    if (a == 1.0) {
      var k = 0
      while (k < nnz) {
        yValues(xIndices(k)) += xValues(k)
        k += 1
      }
    } else {
      var k = 0
      while (k < nnz) {
        yValues(xIndices(k)) += a * xValues(k)
        k += 1
      }
    }
  }

  private def axpy(a: Double, x: DenseVector, y: DenseVector): Unit = {
    val n = x.size
    f2jBLAS.daxpy(n, a, x.values, 1, y.values, 1)
  }

  /**
   * x = a * x
   */
  def scal(a: Double, x: Vector): Unit = {
    x match {
      case sx: SparseVector =>
        f2jBLAS.dscal(sx.values.length, a, sx.values, 1)
      case dx: DenseVector =>
        f2jBLAS.dscal(dx.values.length, a, dx.values, 1)
      case _ =>
        throw new IllegalArgumentException(s"scal doesn't support vector type ${x.getClass}.")
    }
  }
}
