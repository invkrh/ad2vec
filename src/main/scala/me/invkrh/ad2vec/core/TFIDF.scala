package me.invkrh.ad2vec.core

import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel, HashingTF, IDF}
import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vector}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.functions.udf

import me.invkrh.ad2vec.util.Hashing.hashingStr

trait TFIDF {

  def inputCol: String
  def outputCol: String

  def vocabSize: Int
  def term2index(term: String): Int
  protected def termFrequency(docs: DataFrame): DataFrame

//  def docs: DataFrame
//  require(
//    !docs.schema.fieldNames.contains("tf") &&
//      !docs.schema.fieldNames.contains("tf_normalized"),
//    "[tf] and [tf_normalized] are internal col name used in TFIDF"
//  )

  // TODO: Add more normalization method
  private val norm = udf { v: Vector =>
    v match {
      case sv: SparseVector =>
        val total = sv.values.sum
        new SparseVector(sv.size, sv.indices, sv.values.map(_ / total))
      case dv: DenseVector =>
        val total = dv.values.sum
        new DenseVector(dv.values.map(_ / total))
    }
  }

  def transform(docs: DataFrame): DataFrame = {
    val tf = termFrequency(docs).withColumn("tf_normalized", norm(col("tf")))
    val idf = new IDF().setInputCol("tf_normalized").setOutputCol(outputCol)

    idf
      .fit(tf) // compute IDF
      .transform(tf) // compute TFIDF
      .drop("tf", "tf_normalized") // clean column
  }
}

class CntVecTFIDF(
  val inputCol: String,
  val outputCol: String
) extends TFIDF {

  private var termIndex: Option[Map[String, Int]] = None

  def vocabSize: Int = if (termIndex.isEmpty) 0 else termIndex.get.size

  def term2index(term: String): Int = {
    termIndex match {
      case Some(index) => index(term)
      case None =>
        throw new IllegalAccessException(
          "term2index map is created only after tfidf is computed")
    }
  }

  protected def termFrequency(docs: DataFrame): DataFrame = {

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
  val outputCol: String
) extends TFIDF {

  private val hashingTF =
    new HashingTF().setInputCol(inputCol).setOutputCol("tf")

  def vocabSize: Int = hashingTF.getNumFeatures

  /**
   * Hash a term to an index (rely on HashingTF)
   */
  def term2index(term: String): Int = {
    hashingStr(term, hashingTF.getNumFeatures)
  }

  protected def termFrequency(docs: DataFrame): DataFrame =
    hashingTF.transform(docs)
}
