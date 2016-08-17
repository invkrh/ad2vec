package me.invkrh.ad2vec.core

import org.apache.spark.ml.feature.{StopWordsRemover, Tokenizer, Word2Vec, Word2VecModel}
import org.apache.spark.ml.linalg.{SparseVector, Vector, Vectors}
import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.sql.functions.{udf, _}

import me.invkrh.ad2vec.algebra.BLAS

/**
 * TODO:
 * 1. efficient cache data
 * 2. use data pipeline
 * 3. look for dataset conversion for typed code
 * 4. Enumerate language
 * 5. Improve col name
 */
trait Ad2VecBase {
  var idCol: String = "id"
  var docCol: String = "doc"
  var language: String = "english"

  def setIdCol(idCol: String): this.type = {
    this.idCol = idCol
    this
  }

  def setDocCol(docCol: String): this.type = {
    this.docCol = docCol
    this
  }

  def setLanguage(language: String): this.type = {
    this.language = language
    this
  }

  def copyParamsOf[T <: Ad2VecBase](that: T): Unit = {
    that.setIdCol(this.idCol)
    that.setDocCol(this.docCol)
    that.setLanguage(this.language)
  }

  def textProcessing(corpus: DataFrame,
                     removeStopWords: Boolean,
                     replaceNum: Boolean = false): DataFrame = {

    import corpus.sqlContext.implicits._

    /**
     * Remove punctuation
     */
    val punctuationRemoved =
      corpus.select(col(idCol), regexp_replace(col(docCol), "\\p{P}", "").as(docCol))

    /**
     * Tokenize
     */
    val tokenizer =
      new Tokenizer().setInputCol(docCol).setOutputCol("tokens")
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
    val stopWordsRemoved = if (removeStopWords) {
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
    val replaceSingleNum = udf { (xs: Seq[String]) =>
      xs.map(x => if (x.matches("\\d+")) "NUM" else x)
    }
    if (replaceNum) {
      stopWordsRemoved.select(col(idCol), replaceSingleNum($"words").as("words"))
    } else {
      stopWordsRemoved
    }
  }
}

class Ad2Vec(val word2vec: Word2Vec) extends Ad2VecBase {
  def fit(dataSet: DataFrame): Ad2VecModel = {
    val processed =
      textProcessing(dataSet, removeStopWords = false, replaceNum = false)
    val model = new Ad2VecModel(word2vec.fit(processed))
    model.copyParamsOf(this)
    model
  }
}

private[core] class Ad2VecModel(model: Word2VecModel) extends Ad2VecBase {

  def average(weightedDoc: DataFrame, dict: Map[String, (Vector, Int)], dim: Int): DataFrame = {
    val bDict = weightedDoc.sparkSession.sparkContext.broadcast(dict)
    val word2VecTFIDF = udf { (sentence: Seq[String], weights: Vector) =>
      if (sentence.isEmpty) {
        Vectors.sparse(dim, Array.empty[Int], Array.empty[Double])
      } else {
        val sum = Vectors.zeros(dim)
        sentence.foreach { word =>
          bDict.value.get(word).foreach {
            case (v, index) =>
              BLAS.axpy(weights(index), v, sum)
          }
        }
        val wSum = weights.asInstanceOf[SparseVector].values.sum
        if (wSum == 0) {
          BLAS.scal(1.0 / sentence.size, sum)
        } else {
          BLAS.scal(1.0 / wSum, sum)
        }
        sum
      }
    }
    weightedDoc.withColumn("vec", word2VecTFIDF(col("words"), col("tfidf")))
  }

  /**
   * Transform a document column to a vector column to represent the whole document.
   * The transform is performed by averaging all word vectors by the words' tfidf weight
   * in the document.
   *
   * @param raw raw documents data set
   * @return TFIDF weight-averaged document representation
   */
  def transform(raw: DataFrame, tfidf: Option[TFIDF] = None): DataFrame = {
    val processed =
      textProcessing(raw, removeStopWords = true, replaceNum = false)
    tfidf match {
      case Some(x) =>
        // Compute TFIDF weights
        val weightedDoc = x.transform(processed)
        val dict = model.getVectors.rdd.map {
          case Row(word: String, vec: Vector) =>
            (word, (vec, x.term2index(word)))
        }.collect.toMap
        val d = model.getVectorSize
        average(weightedDoc, dict, d)
      case None =>
        model.transform(processed)
    }
  }

}
