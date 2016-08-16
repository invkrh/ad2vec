package me.invkrh.ad2vec.algebra

import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vectors}
import org.scalatest.FunSuite

class BLASTest extends FunSuite {

  test("testScal for dense vector") {
    val x = new DenseVector(Array(1, 2, 3, 4, 5))
    val expected = new DenseVector(Array(2, 4, 6, 8, 10))
    assertResult(expected) {
      BLAS.scal(2, x)
      x
    }
  }

  test("testAxpy for dense vector") {
    val y = Vectors.zeros(5) // DenseVector
    val x = new DenseVector(Array(1, 2, 3, 4, 5))
    val expected = new DenseVector(Array(2, 4, 6, 8, 10))
    assertResult(expected) {
      BLAS.axpy(2, x, y)
      y
    }
  }
  
  test("testScal for sparse vector") {
    val x = new SparseVector(5, Array(0, 1, 2, 3, 4), Array(1, 2, 3, 4, 5))
    val expected = new SparseVector(5, Array(0, 1, 2, 3, 4), Array(2, 4, 6, 8, 10))
    assertResult(expected) {
      BLAS.scal(2, x)
      x
    }
  }
  
  test("testAxpy for sparse vector") {
    val y = Vectors.zeros(5) // DenseVector
    val x = new SparseVector(5, Array(0, 1, 2, 3, 4), Array(1, 2, 3, 4, 5))
    val expected = new SparseVector(5, Array(0, 1, 2, 3, 4), Array(2, 4, 6, 8, 10))
    assertResult(expected) {
      BLAS.axpy(2, x, y)
      y
    }
  }

}
