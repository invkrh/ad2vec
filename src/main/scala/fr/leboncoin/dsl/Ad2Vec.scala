package fr.leboncoin.dsl

import fr.leboncoin.dsl.common._
import org.apache.spark.ml.feature.{StopWordsRemover, Tokenizer, Word2Vec}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._

class Ad2Vec(rawDataSetURL: String) {

  import Ad2Vec._
  import sqlContext.implicits._

  def loadDataSet(removeStopWords: Boolean, replaceNum: Boolean): DataFrame = {
    // Read raw data from S3 (decompression gz)
    val raw = sqlContext.read.json(rawDataSetURL)

    // Remove punctuation
    val puncRemoved = raw.select($"ad_id", regexp_replace($"body", "\\p{P}", "").as("body"))

    // Tokenize
    val tokenizer = new Tokenizer()
      .setInputCol("body")
      .setOutputCol("raw_words")
    val tokenized = tokenizer.transform(puncRemoved)

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
          .setStopWords(stopWordsFR)
          .setInputCol("raw_words")
          .setOutputCol("words_with_empty")
        remover.transform(tokenized)
      } else {
        tokenized.withColumnRenamed("raw_words", "words_with_empty")
      }

    val emptyRemoved =
      stopWordsRemoved.select(
        $"ad_id",
        rmEmptyUDF($"words_with_empty").as("words_without_empty"))

    /**
     * It also might be better not to remove numbers.
     */
    if (replaceNum)
      emptyRemoved.select(
        $"ad_id",
        replaceSingleNum($"words_without_empty").as("words"))
    else
      emptyRemoved.withColumnRenamed("words_without_empty", "words")
  }

  def run() = {
    val dataSet = loadDataSet(removeStopWords = false, replaceNum = false)
    val word2Vec = new Word2Vec()
      .setInputCol("words")
      .setOutputCol("result")
      .setVectorSize(100)
      .setMinCount(5)
      .setNumPartitions(20)
      .setStepSize(0.25)
      .setWindowSize(5)
      .setMaxIter(1)
    val model = word2Vec.fit(dataSet)
    model.transform(loadDataSet(removeStopWords = true, replaceNum = false))
  }
}

object Ad2Vec {
  val rmEmptyUDF =
    sqlContext.udf.register("removeEmpty", (xs: Seq[String]) => xs.filter(_.nonEmpty))

  val replaceSingleNum =
    sqlContext.udf.register("removeEmpty", (xs: Seq[String]) =>
      xs.map(x => if (x.matches("\\d+")) "NUM" else x))

  lazy val stopWordsFR = {
    val is = getClass.getResourceAsStream(s"/stopwords/french.txt")
    scala.io.Source.fromInputStream(is)(scala.io.Codec.UTF8).getLines().toArray
  }

  def main(args: Array[String]) {
    require(args.length == 1)
    val rawDataSetURL = args.head
    new Ad2Vec(rawDataSetURL).run()
  }
}
