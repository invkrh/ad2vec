package me.invkrh.ad2vec

import org.apache.spark.sql.SparkSession

package object core {
  val spark = SparkSession.builder().appName("ad2vec").master("local[*]").getOrCreate()
}
