package fr.leboncoin.ad2vec

import org.apache.spark.ml.feature.Word2VecModel
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.unsafe.hash.Murmur3_x86_32._
import org.apache.spark.unsafe.types.UTF8String

class A(val value: Int)

class Wrapper() extends Serializable {
  @transient lazy val a = {
    println("A is initialized")
    new A(1)
  }
}

object TestRDD extends App {
  val spark = SparkSession
    .builder()
    .appName("ad2vec")
    .master("local[*]")
    .getOrCreate()

  def run() = {
    val w = new Wrapper()
    //    val v = w.a.value
    val rdd = spark.sparkContext.makeRDD(1 to 100, 8)
    val res = rdd.map(w.a.value + _).count()

    println(res)
  }

  run()
}


class Testable {
  def func(rdd: RDD[(Int, Int)]): Unit = {
    def one(a: Int, b: Int) = a + b
    rdd.map {
      case (key, value) => value + one(1, 2)
    }.count
  }

//  def func(df: DataFrame): Unit = {
//    import org.apache.spark.sql.functions.udf
//    import spark.sqlContext.implicits._
//    def one = 1
//    val add = udf {
//      (a: Int) => a + one
//    }
//    val result = df.withColumn("new", add($"value"))
//    result.show() // It should not work, but it works
//    result.filter("key = 2").show
//  }
}

object DataFrameSerDeTest extends App {
  val spark = SparkSession
    .builder()
    .appName("test")
    .master("local[*]")
    .getOrCreate()

  class A(val value: Int)

  val df = spark.createDataFrame(Seq(
    (1, 2),
    (2, 2),
    (3, 2),
    (4, 2)
  )).toDF("key", "value")

  val rdd = spark.sparkContext.makeRDD(Seq(
    (1, 2),
    (2, 2),
    (3, 2),
    (4, 2)
  ))

  def run() = {
    // new Testable().func(df)
    new Testable().func(rdd)
  }

  //  def run() = {
  //    new Testable().run(df)
  //  }

  //  def run() = {
  //    import org.apache.spark.sql.functions.udf
  //    import spark.sqlContext.implicits._
  //
  //    val notSer = new A(2)
  //
  //    //    val add = udf {
  //    //      (a: Int) => a + notSer.value
  //    //    }
  //    //
  //    //    val added = df.select($"key", add($"value").as("added"))
  //    //    added.show()
  //    //    added.filter($"key" === 2).show()
  //  }

  run()
}
