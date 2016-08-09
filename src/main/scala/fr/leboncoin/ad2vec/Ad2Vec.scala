package fr.leboncoin.ad2vec

import org.apache.spark.ml.feature.{StopWordsRemover, Tokenizer, Word2Vec, Word2VecModel}
import org.apache.spark.ml.linalg.{SparseVector, Vector, Vectors}
import org.apache.spark.ml.param.{Param, Params}
import org.apache.spark.sql.functions.{udf, _}
import org.apache.spark.sql.{DataFrame, Row, SaveMode}
import org.apache.spark.unsafe.hash.Murmur3_x86_32._
import org.apache.spark.unsafe.types.UTF8String

/**
 * TODO:
 * 1. efficient cache data
 * 3. look for dataset conversion for typed code
 */

abstract class Ad2Vec private(
  private var idCol: String,
  private var docCol: String,
  private var language: String,
  private var weighted: Boolean) {

  def this() = this("id", "doc", "english", false)

  // Set word2Vec parameters while creating the Ad2Vec
  val word2Vec: Word2Vec

  def setIdCol(idCol: String) = {
    this.idCol = idCol
    this
  }

  def setDocCol(docCol: String) = {
    this.docCol = docCol
    this
  }

  def setLanguage(language: String) = {
    this.language = language
    this
  }

  def setWeighted(weighted: Boolean) = {
    this.weighted = weighted
    this
  }

  import sqlContext.implicits._

  def textProcessing(
    corpus: DataFrame,
    removeStopWords: Boolean,
    replaceNum: Boolean = false): DataFrame = {

    /**
     * Remove punctuation
     */
    val punctuationRemoved =
      corpus
        .select(col(idCol), regexp_replace(col(docCol), "\\p{P}", "").as(docCol))

    /**
     * Tokenize
     */
    val tokenizer = new Tokenizer()
      .setInputCol(docCol)
      .setOutputCol("tokens")
    val tokenized = tokenizer.transform(punctuationRemoved)

    /**
     * Stemming can be complex, candidate: snowball, lucene
     * Checkout maven repo for available jars
     */

    /**
     * Remove stopwords
     * Note: train Word2Vec it is better not to remove stop words because the algorithm relies on
     * the broader context of the sentence in order to produce high-quality word vectors.
     * For this reason, we will make stop word removal optional in the functions.
     */
    val stopWordsRemoved =
      if (removeStopWords) {
        val remover = new StopWordsRemover()
          .setStopWords(StopWordsRemover.loadDefaultStopWords(language))
          .setInputCol("tokens")
          .setOutputCol("words")
        remover.transform(tokenized)
      } else {
        tokenized.withColumnRenamed("tokens", "words")
      }

    /**
     * It also might be better not to remove numbers.
     */
    val replaceSingleNum = udf {
      (xs: Seq[String]) => xs.map(x => if (x.matches("\\d+")) "NUM" else x)
    }
    if (replaceNum)
      stopWordsRemoved.select(col(idCol), replaceSingleNum($"words").as("words"))
    else
      stopWordsRemoved
  }

  def fit(dataSet: DataFrame): Word2VecModel = {
    word2Vec.fit(dataSet)
  }

  def computeWeight(processed: DataFrame) = {
    val tfidf = new HashingTFIDF("words", "tfidf", processed)
    tfidf.result()
  }

  def weightedAverageDocVec(weightedDoc: DataFrame, wordVecIndex: Map[String, Vector]) = {
    // Create udf for doc vec
    val bWordVecIndex = weightedDoc.sparkSession.sparkContext.broadcast(wordVecIndex)
    val d = word2Vec.getVectorSize
    val word2VecTFIDF = udf {
      (sentence: Seq[String], weights: Vector) =>
        if (sentence.isEmpty) {
          Vectors.sparse(d, Array.empty[Int], Array.empty[Double])
        } else {
          val sum = Vectors.zeros(d)
          sentence.foreach { word =>
            val i = Ad2Vec.term2index(word)
            val w = weights(i)
            bWordVecIndex.value.get(word).foreach { v =>
              axpy(w, v, sum)
            }
          }
          val weightSum = weights.asInstanceOf[SparseVector].values.sum
          if (weightSum != 0) {
            scal(1.0 / weightSum, sum)
          } else {
            scal(1.0 / sentence.size, sum)
          }
          sum
        }
    }

    // Weighted averaged word vectors by their tfidf
    weightedDoc
      .withColumn("vec", word2VecTFIDF(col("words"), col("tfidf")))
  }

  implicit class Word2VecModelHolder(model: Word2VecModel) {
    def transformWithTFIDF(processed: DataFrame) = {
      val index = model.getVectors.rdd.map {
        case Row(word: String, vec: Vector) => (word, vec)
      }.collect.toMap
      val weighted = computeWeight(processed)
      weightedAverageDocVec(weighted, index)
    }
  }

  def transform(corpus: DataFrame, modelURL: Option[String] = None): DataFrame = {
    val withStopWords = textProcessing(corpus, removeStopWords = false)
    val model = this.fit(withStopWords)
    modelURL.foreach(url => model.save(url + s"/${model.uid}"))
    val noStopWords = textProcessing(corpus, removeStopWords = true)
    if (weighted) {
      model.transformWithTFIDF(noStopWords).select(idCol, "vec")
    } else {
      model.transform(noStopWords).select(idCol, "vec")
    }
  }
}

object Ad2Vec {

  def term2index(term: String): Int = {
    val seed = 42
    val utf8 = UTF8String.fromString(term)
    val hashNum = hashUnsafeBytes(utf8.getBaseObject, utf8.getBaseOffset, utf8.numBytes(), seed)
    nonNegativeMod(hashNum, 1 << 18)
  }

  def main(args: Array[String]): Unit = {
    /**
     * Convert json file to parquet file
     */
    def convertCorpusIntoParquet(rawURL: String, corpusURL: String): Unit = {
      // TODO: add an option to divide ads into sentences
      sqlContext.read.json(rawURL)
        .select("ad_id", "body")
        .write.mode(SaveMode.Overwrite)
        .parquet(corpusURL)
    }

    def result(
      corpusURL: String,
      modelURL: Option[String] = None,
      resultURL: Option[String] = None): DataFrame = {

      /**
       * Read corpus (parquet file) from S3
       */
      val raw = sqlContext.read.parquet(corpusURL)
      val resDF = new Ad2Vec() {
        override val word2Vec: Word2Vec = new Word2Vec
      }.transform(raw, modelURL)
      resultURL foreach resDF.write.mode(SaveMode.Overwrite).parquet
      resDF
    }
  }
}
