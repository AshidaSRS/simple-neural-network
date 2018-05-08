import scala.util.Random
import breeze.linalg._
import breeze.numerics._

object MainApp extends App {

  val set1 = DenseVector(0,0,1)
  val set2 = DenseVector(1,1,1)
  val set3 = DenseVector(1,0,1)
  val set4 = DenseVector(0,1,1)
  val outSet1 = DenseVector(0, 1, 1, 0)

  val trainingSetInputs = DenseMatrix(set1, set2, set3, set4)
  val trainingSetOutputs = DenseMatrix(outSet1).t

  def serial(): Unit = Random.setSeed(1)

  def getRandom = 2 * Random.nextDouble() -1
  var weights: DenseMatrix[Double] = DenseMatrix.create(3, 1, Array(getRandom, getRandom, getRandom))
  println(s"First weigths \n$weights")

  def think(inputs: DenseMatrix[Int]): DenseMatrix[Double] = sigmoid(convert(inputs, Double) * weights)

  def sigmoidDerivative(x: DenseMatrix[Double]): DenseMatrix[Double] = sigmoid(x) *:* (-sigmoid(x) + 1.0)

  def train(inputs: DenseMatrix[Int], outputs: DenseMatrix[Int]): Unit = {

    def trainAux(inputs: DenseMatrix[Int], outputs: DenseMatrix[Int], n: Int): DenseMatrix[Double] = {
      n match {
        case 0 => weights
        case _ =>
          val output: DenseMatrix[Double] = think(inputs)
          val error: DenseMatrix[Double] = convert(outputs, Double) - output
          val adjust: DenseMatrix[Double] = convert(inputs, Double).t * (error *:* sigmoidDerivative(output))
          weights = weights + adjust
          trainAux(inputs, outputs, n - 1)
      }
    }

    trainAux(inputs, outputs, 10000)
  }
  train(trainingSetInputs, trainingSetOutputs)

  println("Last weights")
  println(weights)

  println(s"New situation [1,0,0]: ")
  println(think(DenseMatrix.create(1,3, Array(1,0,0))))
}