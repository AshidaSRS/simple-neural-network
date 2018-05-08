import scala.util.Random
import breeze.linalg._
import breeze.math._
import breeze.numerics._

object MainApp extends App {

  val set1 = DenseVector(0,0,1)
  val set2 = DenseVector(1,1,1)
  val set3 = DenseVector(1,0,1)
  val set4 = DenseVector(0,1,1)
  val outSet1 = DenseVector(0, 1, 1, 0)

  val trainingSetInputs = DenseMatrix(set1, set2, set3, set4)
  val trainingSetOutputs = DenseMatrix(outSet1).t
  println(trainingSetInputs)
  println(trainingSetOutputs)
  println("-----------")

  def serial(): Unit = Random.setSeed(1)

  def getRandom = 2 * Random.nextDouble() -1
  var weights: DenseMatrix[Double] = DenseMatrix.create(3, 1, Array(getRandom, getRandom, getRandom))
  println(weights)
  println("-----------")
  def think(inputs: DenseMatrix[Int]): DenseMatrix[Int] = convert(sigmoid(convert(inputs, Double) * weights), Int)

  def sigmoidDerivative(x: DenseMatrix[Int]): DenseMatrix[Int] = {
    println(x)
    val a = x * (1 -:- x)
    println(a)
    a
  }

  def train(inputs: DenseMatrix[Int], outputs: DenseMatrix[Int]): Unit = {

    def trainAux(inputs: DenseMatrix[Int], outputs: DenseMatrix[Int], n: Int): DenseMatrix[Double] = {
      n match {
        case 0 => weights
        case _ =>
          val output: DenseMatrix[Int] = think(inputs)
          val error: DenseMatrix[Double] = convert(outputs, Double) - convert(output, Double)
          val adjust: DenseMatrix[Double] = convert(inputs, Double).t * (error * convert(sigmoidDerivative(output), Double))
          weights += adjust
          trainAux(inputs, outputs, n - 1)
      }
    }

    trainAux(inputs, outputs, 10000)
  }
  println("-----------")
  train(trainingSetInputs, trainingSetOutputs)
  println(weights)
  println(think(DenseMatrix.create(1,3, Array(1,0,0))))
}