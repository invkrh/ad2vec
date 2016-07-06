package fr.leboncoin.dsl

import common._
import org.apache.spark.mllib.feature.Word2Vec
import org.apache.spark.rdd.RDD

class Ad2vec(rawDataSetURL: String) {

  def preprocess(): RDD[Seq[String]] = {
    val df = sqlContext.read.json(rawDataSetURL)
    // TODO: tokenizer, stemming, stopwords, etc
    ???
  }

  def run() = {
    val dataSet = preprocess()
    val word2Vec = new Word2Vec()
      .setNumPartitions(20)
      .setWindowSize(5)
    word2Vec.fit(dataSet)
  }
}

object Ad2vec {
  def main(args: Array[String]) {
    require(args.length == 1)
    val rawDataSetURL = args.head
    val task = new Ad2vec(rawDataSetURL)
    task.preprocess()
  }
}
