package fr.leboncoin.dsl

import com.amazonaws.auth.profile.{ProfilesConfigFile, ProfileCredentialsProvider}
import com.amazonaws.auth.{InstanceProfileCredentialsProvider, SystemPropertiesCredentialsProvider, EnvironmentVariableCredentialsProvider, AWSCredentialsProviderChain}
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}

package object common {
  val sc         = new SparkContext("local[*]", "ad2vec", new SparkConf)
  val sqlContext = new SQLContext(sc)

  // AWS credential
  val chain =
    new AWSCredentialsProviderChain(
      new EnvironmentVariableCredentialsProvider,
      new SystemPropertiesCredentialsProvider,
      new InstanceProfileCredentialsProvider)
  sc.hadoopConfiguration.set("fs.s3a.access.key", chain.getCredentials.getAWSAccessKeyId)
  sc.hadoopConfiguration.set("fs.s3a.secret.key", chain.getCredentials.getAWSSecretKey)
}
