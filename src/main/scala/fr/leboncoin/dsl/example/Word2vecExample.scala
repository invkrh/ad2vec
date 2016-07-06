package fr.leboncoin.dsl.example

import fr.leboncoin.dsl.common._
import org.apache.spark.mllib.feature.{Word2Vec, Word2VecModel}

object Word2vecExample extends App {

  val input = sc.textFile("/home/invkrh/dataset/text8")
    .map(line => line.split(" ").toSeq)

  val word2vec = new Word2Vec().setNumPartitions(20)

  val model = word2vec.fit(input)

  val synonyms = model.findSynonyms("china", 40)

  for ((synonym, cosineSimilarity) <- synonyms) {
    println(s"$synonym $cosineSimilarity")
  }

  // Save and load model
  model.save(sc, "model")
  val sameModel = Word2VecModel.load(sc, "model")
}
