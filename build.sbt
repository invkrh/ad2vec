name := "ad2vec"

version := "0.0.1"

scalaVersion := "2.11.8"

libraryDependencies += "org.apache.spark" %% "spark-mllib" % "1.6.2"

libraryDependencies += "org.apache.hadoop" % "hadoop-aws" % "2.7.2" excludeAll ExclusionRule(organization = "javax.servlet")

libraryDependencies += "org.scalactic" %% "scalactic" % "2.2.6"

libraryDependencies += "org.scalatest" %% "scalatest" % "2.2.6" % "test"
