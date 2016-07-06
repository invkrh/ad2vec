package fr.leboncoin.dsl

import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}

package object common {
  val sc         = new SparkContext("local[*]", "ad2vec", new SparkConf)
  //  val hiveContext = new HiveContext(sc)
  val sqlContext = new SQLContext(sc)
}
