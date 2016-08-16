package me.invkrh.ad2vec.util

import org.apache.spark.unsafe.hash.Murmur3_x86_32._
import org.apache.spark.unsafe.types.UTF8String

object Hashing {
  def nonNegativeMod(x: Int, mod: Int): Int = {
    val rawMod = x % mod
    rawMod + (if (rawMod < 0) mod else 0)
  }

  def hashingStr(term: String, nbBucket: Int): Int = {
    val seed = 42
    val utf8 = UTF8String.fromString(term)
    val hashNum = hashUnsafeBytes(utf8.getBaseObject, utf8.getBaseOffset, utf8.numBytes(), seed)
    nonNegativeMod(hashNum, nbBucket)
  }
}
