package fr.leboncoin.dsl

import fr.leboncoin.dsl.common._
import org.apache.spark.ml.feature.{StopWordsRemover, Tokenizer, Word2Vec}
import org.apache.spark.sql.{SaveMode, DataFrame}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.functions.udf

class Ad2Vec(corpusURL: String) {

  import Ad2Vec._
  import sqlContext.implicits._

  //TODO: efficient cache data
  def loadCorpus(removeStopWords: Boolean, replaceNum: Boolean): DataFrame = {

    /**
     * Read corpus (parquet file) from S3
     */
    val raw = sqlContext.read.parquet(corpusURL)

    /**
     * Remove punctuation
     */
    val punctuationRemoved =
      raw.select($"ad_id", regexp_replace($"body", "\\p{P}", "").as("body"))

    /**
     * Tokenize
     */
    val tokenizer = new Tokenizer().setInputCol("body").setOutputCol("tokens")
    val tokenized = tokenizer.transform(punctuationRemoved)

    /**
     * Stemming can be complex, candidate: snowball, lucene
     * Checkout maven repo for available jars
     */
    /**
     * Remove stopwords
     * Note: when training Word2Vec, it is better not to remove stop words
     * because the algorithm relies on the broader context of the sentence
     * in order to produce high-quality word vectors. For this reason,
     * we will make stop word removal optional in the functions.
     */
    val stopWordsRemoved = if (removeStopWords) {
      val remover = new StopWordsRemover()
        .setStopWords(stopWordsFR)
        .setInputCol("tokens")
        .setOutputCol("words")
      remover.transform(tokenized)
    } else {
      tokenized.withColumnRenamed("tokens", "words")
    }

    val emptyRemoved =
      stopWordsRemoved.select($"ad_id", rmEmptyUDF($"words").as("words"))

    /**
     * It also might be better not to remove numbers.
     */
    if (replaceNum)
      emptyRemoved.select($"ad_id", replaceSingleNum($"words").as("words"))
    else
      emptyRemoved
  }

  def run(modelURL: Option[String] = None): DataFrame = {
    val dataSet = loadCorpus(removeStopWords = false, replaceNum = false)
    val word2Vec = new Word2Vec()
      .setInputCol("words")
      .setOutputCol("vec")
      .setVectorSize(100)
      .setMinCount(5)
      .setNumPartitions(20)
      .setStepSize(0.25)
      .setWindowSize(5)
      .setMaxIter(1)
    val model = word2Vec.fit(dataSet)
    modelURL.foreach(url => model.save(url + s"/${model.uid}"))
    model
      .transform(loadCorpus(removeStopWords = true, replaceNum = false))
      .select("ad_id", "vec")
  }
}

object Ad2Vec {

  val rmEmptyUDF = udf { (xs: Seq[String]) =>
    xs.filter(_.nonEmpty)
  }

  val replaceSingleNum = udf { (xs: Seq[String]) =>
    xs.map(x => if (x.matches("\\d+")) "NUM" else x)
  }

  lazy val stopWordsFR: Array[String] = {
    val is = getClass.getResourceAsStream(s"/stopwords/french.txt")
    scala.io.Source.fromInputStream(is)(scala.io.Codec.UTF8).getLines().toArray
  }

  def createCorpus(rawURL: String, corpusURL: String): Unit = {
    sqlContext.read
      .json(rawURL)
      .select("ad_id", "body")
      .write
      .mode(SaveMode.Overwrite)
      .parquet(corpusURL)
  }

  def result(corpusURL: String,
             modelURL: Option[String] = None,
             resultURL: Option[String] = None): DataFrame = {
    val resDF = new Ad2Vec(corpusURL).run(modelURL)
    resultURL foreach resDF.write.mode(SaveMode.Overwrite).parquet
    resDF
  }

  def main(args: Array[String]) {}
}
