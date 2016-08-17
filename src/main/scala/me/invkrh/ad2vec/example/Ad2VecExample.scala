package me.invkrh.ad2vec.example

import com.amazonaws.auth.{
  AWSCredentialsProviderChain,
  EnvironmentVariableCredentialsProvider,
  SystemPropertiesCredentialsProvider
}
import org.apache.spark.ml.feature.Word2Vec
import org.apache.spark.sql.{SaveMode, _}

import me.invkrh.ad2vec.core.Ad2Vec

object Ad2VecExample {

  val spark = SparkSession.builder().appName("ad2vec").master("local[*]").getOrCreate()

  /**
   * Efficient parquet committer
   */
  spark.sparkContext.hadoopConfiguration.set(
    "spark.sql.parquet.output.committer.class",
    "org.apache.spark.sql.parquet.DirectParquetOutputCommitter"
  )

  /**
   * Load AWS credential
   */
  val chain = new AWSCredentialsProviderChain(
    new EnvironmentVariableCredentialsProvider,
    new SystemPropertiesCredentialsProvider
  )
  spark.sparkContext.hadoopConfiguration
    .set("fs.s3a.access.key", chain.getCredentials.getAWSAccessKeyId)
  spark.sparkContext.hadoopConfiguration
    .set("fs.s3a.secret.key", chain.getCredentials.getAWSSecretKey)

  /**
   * Convert json file to parquet file
   */
  def convertCorpusIntoParquet(rawURL: String, corpusURL: String): Unit = {
    // TODO: add an option to divide ads into sentences
    spark.sqlContext.read
      .json(rawURL)
      .select("ad_id", "body")
      .write
      .mode(SaveMode.Overwrite)
      .parquet(corpusURL)
  }

  def result(corpusURL: String,
             modelURL: Option[String] = None,
             resultURL: Option[String] = None): DataFrame = {

    /**
     * Read corpus (parquet file) from S3
     */
    val raw = spark.sqlContext.read.parquet(corpusURL)
    val w2v = new Word2Vec()
      .setInputCol("words")
      .setOutputCol("vec")
      .setVectorSize(100)
      .setMinCount(5)
      .setNumPartitions(20)
      .setStepSize(0.25)
      .setWindowSize(5)
      .setMaxIter(1)
    val ad2vec = new Ad2Vec(w2v).setIdCol("ad_id").setDocCol("body")
    val model = ad2vec.fit(raw)
    val resDF = model.transform(raw)

    resultURL foreach resDF.write.mode(SaveMode.Overwrite).parquet
    resDF
  }

  def main(args: Array[String]): Unit = {}
}
