package fr.leboncoin.dsl

import org.scalatest.FunSuite

class Ad2VecTest extends FunSuite {
  val bucket = "data.dev.leboncoin.io-datalake"
  val prefix = "data-science-lab/word2vec/raw/blocket/clothes"
  val file = s"s3a://$bucket/$prefix/20160429.json.gz"
  test("Ad2vec should read s3 file") {
    val job = new Ad2Vec(file)
    val ds = job.loadDataSet(removeStopWords = false, replaceNum = false)
    ds.take(100) foreach println
    ds.count()
  }

}
