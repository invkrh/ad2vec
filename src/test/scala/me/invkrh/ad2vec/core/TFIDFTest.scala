package me.invkrh.ad2vec.core

import scala.collection.mutable
import scala.math.log

import org.apache.spark.ml.feature.Tokenizer
import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.sql.functions.udf
import org.scalatest.{BeforeAndAfterEach, FunSuite}


class TFIDFTest extends FunSuite with BeforeAndAfterEach {

  val testData = {
    val sentence = spark.createDataFrame(Seq(
      (0, "A B C"),
      (1, "B C D"),
      (2, "A B D")
    )).toDF("id", "body")
    val tokenizer = new Tokenizer().setInputCol("body").setOutputCol("words")
    tokenizer.transform(sentence)
  }

  def testCase(tfidf: TFIDF, dataset: DataFrame): Unit = {
    val termValueMap = term2Value(tfidf, dataset)
    val expected = 1d / 3 * log(4d / 3)
    assert(termValueMap("A") === expected)
    assert(termValueMap("B") === 0d)
    assert(termValueMap("C") === expected)
    assert(termValueMap("D") === expected)
  }

  def time[T](thunk: => T): (T, Long) = {
    val start = System.currentTimeMillis()
    val res = thunk
    val elapsedTime = System.currentTimeMillis() - start
    // scalastyle:off println
    println("Time used: " + elapsedTime + " ms")
    // scalastyle:on println
    (res, elapsedTime)
  }

  def term2Value(tfidf: TFIDF, dataset: DataFrame): (String) => Double = {
    import dataset.sqlContext.implicits._
    val result = tfidf.transform(dataset)
    val getIndex = udf { v: SparseVector => v.indices }
    val getValue = udf { v: SparseVector => v.values }
    val index2value = result.select(getIndex($"TFIDF"), getValue($"TFIDF"))
      .rdd
      .flatMap { r: Row =>
        r.getAs[mutable.WrappedArray[Int]](0)
          .zip(r.getAs[mutable.WrappedArray[Double]](1))
      }.collectAsMap()
    ((s: String) => s.toLowerCase)
      .andThen(tfidf.term2index)
      .andThen(index2value)
  }

  //  val epsilon = 1e-5f
  //  implicit val doubleEq = TolerantNumerics.tolerantDoubleEquality(epsilon)

  test("TFIDF with HashingTF") {
    val tfidf = new HashingTFIDF("words", "TFIDF")
    testCase(tfidf, testData)
  }

  test("TFIDF with CountVectorizer") {
    val tfidf = new CntVecTFIDF("words", "TFIDF")
    testCase(tfidf, testData)
  }
}
