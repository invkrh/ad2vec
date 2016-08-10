package me.invkrh.ad2vec.core

import org.apache.spark.ml.feature.{StopWordsRemover, Word2Vec}
import org.apache.spark.ml.linalg.{DenseVector, SparseVector}
import org.apache.spark.sql.{Row, SparkSession}
import org.scalactic.TolerantNumerics
import org.scalatest.FunSuite

class Ad2VecTest extends FunSuite {
  
  import spark.sqlContext.implicits._

  val word2Vec: Word2Vec = new Word2Vec()
    .setInputCol("words")
    .setOutputCol("vec")
    .setVectorSize(1) // TODO: to adjust
    .setMinCount(1) // TODO: to adjust
    .setNumPartitions(20)
    .setStepSize(0.25)
    .setWindowSize(5)
    .setMaxIter(1)

  test("Text processing should remove punctuation and stop words") {
    val corpus = spark
      .createDataFrame(
        Seq(
          (0, "Glad to meet you spark! You are awesome"),
          (1, "Suicide squad is wonderful"),
          (2, "Harley quinn is so cute")
        ))
      .toDF("id", "doc")
    val ad2vec = new Ad2Vec(word2Vec)
    val df =
      ad2vec.textProcessing(corpus, removeStopWords = true, replaceNum = false)
    val res = df
      .select($"words")
      .rdd
      .map { r =>
        r.getAs[Seq[String]]("words")
      }
      .collect
      .flatten
    assertResult(false) {
      res.exists(_.contains("!"))
    }
    assertResult(false) {
      res.exists(StopWordsRemover.loadDefaultStopWords("english").contains)
    }
  }

  test("Ad2vec with Hashing TFIDF") {

    def hash(term: String): Int = new HashingTFIDF("", "").term2index(term)
    val size = 1 << 18

    val weightedData = Seq(
      (0, Seq("A", "B", "C")), // 0, 0, 0
      (1, Seq("B", "C", "D")), // 1, 2, 3
      (2, Seq("C", "D", "A")) // 2, 4, 6
    ).map {
      case (id, words) =>
        val (indices, values) =
          (words.map(hash).toArray zip Array(id * 1d, id * 2d, id * 3d))
            .sortBy(_._1)
            .unzip // indices should be sorted for SparseVector
        (id, words, new SparseVector(size, indices, values))
    }

    val weighted =
      spark.createDataFrame(weightedData).toDF("id", "words", "tfidf")

    weighted.show(false)

    val dict = Map(
      "A" -> new DenseVector(Array(1d)),
      "B" -> new DenseVector(Array(2d)),
      "C" -> new DenseVector(Array(3d)),
      "D" -> new DenseVector(Array(4d))
    ).map {
      case (w, v) => (w, (v, hash(w)))
    }

    val model = new Ad2VecModel(null)
    val resDF = model.average(weighted, dict, 1).select("id", "vec")

    val groundTruth = Map(
      0 -> new DenseVector(Array(0d)), // when weight sum == 0
      1 -> new DenseVector(Array((1 * 2 + 2 * 3 + 3 * 4) / 6d)),
      2 -> new DenseVector(Array((2 * 3 + 4 * 4 + 6 * 1) / 12d))
    )

    val result = resDF.rdd.map {
      case Row(id: Int, vec: DenseVector) => (id, vec)
    }.collectAsMap()

    groundTruth.foreach {
      case (id, expected) =>
        assertVectorEqual(result(id), expected)
    }
  }

  def assertVectorEqual(vec1: DenseVector, vec2: DenseVector): Unit = {
    val epsilon = 1e-5f
    implicit val doubleEq = TolerantNumerics.tolerantDoubleEquality(epsilon)
    (vec1.toArray zip vec2.toArray) foreach {
      case (v1, v2) => assert(v1 === v2)
    }
  }
}
