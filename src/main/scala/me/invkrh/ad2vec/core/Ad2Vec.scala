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
class Ad2Vec private (private var idCol: String,
                      private var docCol: String,
                      private var language: String,
                      private val word2vec: Word2Vec) {

  def this(word2vec: Word2Vec) = this("id", "doc", "english", word2vec)

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

  def textProcessing(corpus: DataFrame,
                     removeStopWords: Boolean,
                     replaceNum: Boolean = false): DataFrame = {

    import corpus.sqlContext.implicits._

    /**
     * Remove punctuation
     */
    val punctuationRemoved = corpus
      .select(col(idCol), regexp_replace(col(docCol), "\\p{P}", "").as(docCol))

    /**
     * Tokenize
     */
    val tokenizer = new Tokenizer().setInputCol(docCol).setOutputCol("tokens")
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
      stopWordsRemoved
        .select(col(idCol), replaceSingleNum($"words").as("words"))
    } else {
      stopWordsRemoved
    }
  }

  def fit(dataSet: DataFrame): Ad2VecModel = {
    val processed =
      textProcessing(dataSet, removeStopWords = false, replaceNum = false)
    new Ad2VecModel(word2vec.fit(processed))
  }
}

private[core] class Ad2VecModel(model: Word2VecModel) {

  def average(weightedDoc: DataFrame,
              dict: Map[String, (Vector, Int)],
              dim: Int): DataFrame = {
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
   * @param processed documents where all stop words are filtered
   * @return TFIDF weight-averaged document representation
   */
  def transform(processed: DataFrame, tfidf: Option[TFIDF] = None): DataFrame = {
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
