package fr.leboncoin.ad2vec

import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel, HashingTF, IDF}
import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vector}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.udf
import org.apache.spark.unsafe.hash.Murmur3_x86_32.hashUnsafeBytes
import org.apache.spark.unsafe.types.UTF8String

import fr.leboncoin.ad2vec.sqlContext.implicits._

trait TFIDF {

  def inputCol: String
  def outputCol: String
  def docs: DataFrame

  def term2index(term: String): Int
  def termFrequency: DataFrame

  require(
    !docs.schema.fieldNames.contains("tf") &&
      !docs.schema.fieldNames.contains("tf_normalized"),
    "[tf] and [tf_normalized] are internal col name used in TFIDF"
  )

  // TODO: Add more normalization method
  val norm = udf { v: Vector =>
    v match {
      case sv: SparseVector =>
        val total = sv.values.sum
        new SparseVector(sv.size, sv.indices, sv.values.map(_ / total))
      case dv: DenseVector =>
        val total = dv.values.sum
        new DenseVector(dv.values.map(_ / total))
    }
  }

  def result(): DataFrame = {
    val tf = termFrequency.withColumn("tf_normalized", norm($"tf"))
    val idf = new IDF().setInputCol("tf_normalized").setOutputCol(outputCol)

    idf
      .fit(tf) // compute IDF
      .transform(tf) // compute TFIDF
      .drop("tf", "tf_normalized") // clean column
  }
}

class CntVecTFIDF(
  val inputCol: String,
  val outputCol: String,
  val docs: DataFrame
) extends TFIDF {

  private var termIndex: Option[Map[String, Int]] = None

  def term2index(term: String): Int = {
    termIndex match {
      case Some(index) => index(term)
      case None =>
        throw new IllegalAccessException(
          "term2index map is created after term frequency is computed")
    }
  }

  lazy val termFrequency: DataFrame = {

    /**
     * cvModel contains a vocabulary list which may take a lot of memory
     * when the corpus is large
     */
    val cvModel: CountVectorizerModel =
      new CountVectorizer().setInputCol(inputCol).setOutputCol("tf").fit(docs)
    if (termIndex.isEmpty) {
      termIndex = Some(cvModel.vocabulary.view.zipWithIndex.toMap)
    }
    cvModel.transform(docs)
  }
}

class HashingTFIDF(
  val inputCol: String,
  val outputCol: String,
  val docs: DataFrame
) extends TFIDF {

  val hashingTF = new HashingTF().setInputCol(inputCol).setOutputCol("tf")

  /**
   * Hash a term to an index (rely on HashingTF)
   */
  def term2index(term: String): Int = {
    val seed = 42
    val utf8 = UTF8String.fromString(term)
    val hashNum = hashUnsafeBytes(utf8.getBaseObject, utf8.getBaseOffset, utf8.numBytes(), seed)
    nonNegativeMod(hashNum, hashingTF.getNumFeatures)
  }

  @transient lazy val termFrequency: DataFrame = hashingTF.transform(docs)
}
