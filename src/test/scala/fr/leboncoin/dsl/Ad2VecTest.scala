package fr.leboncoin.dsl

import fr.leboncoin.dsl.common._
import org.apache.spark.ml.feature.Word2VecModel
import org.scalatest.FunSuite

class Ad2VecTest extends FunSuite {

  def prefix(tpe: String) = s"data-science-lab/word2vec/blocket/clothes/$tpe"

  val bucket = "data.dev.leboncoin.io-datalake"

  val raw_prefix = prefix("raw")
  val corpus_prefix = prefix("corpus")
  val model_prefix = prefix("model")
  val result_prefix = prefix("result")

  val rawURL = s"s3a://$bucket/$raw_prefix"
  val corpusURL = s"s3a://$bucket/$corpus_prefix"
  val modelURL = s"s3a://$bucket/$model_prefix"
  val resultURL = s"s3a://$bucket/$result_prefix"

  test("Ad2vec should load corpus by need") {
    val df = new Ad2Vec(corpusURL)
      .loadCorpus(removeStopWords = true, replaceNum = true)
    df.take(10) foreach println
  }

  test("Ad2vec should convert json file to parquet file") {
    Ad2Vec.createCorpus(rawURL, corpusURL)
  }

  test("Ad2vec should create result") {
    Ad2Vec.result(corpusURL, Some(modelURL), Some(resultURL))
  }

  test("Checkout result") {
    val df = sqlContext.read.parquet(resultURL)
    df.show(truncate = false)
  }

  test("Model check") {
    init() // init common object in which there is sc
    val model = Word2VecModel.load(s"$modelURL/w2v_db77e6b7a65e")
    val res = model.findSynonyms("dior", 10)
    res.show()
  }

}
