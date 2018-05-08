import scala.util.Random
import breeze.linalg._
import breeze.numerics._

object MainApp extends App {

  val row = 3
  val col = 1
  val learning = 10000

  val set1 = DenseVector(0,0,1)
  val set2 = DenseVector(1,1,1)
  val set3 = DenseVector(1,0,1)
  val set4 = DenseVector(0,1,1)
  val outSet1 = DenseVector(0,1,1,0)
  val outSet2 = DenseVector(1,1,1,1)

  val trainingSetInputs = DenseMatrix(set1, set2, set3, set4)
  val trainingSetOutputs = DenseMatrix(outSet1).t

  println(s"Train entry: \n$trainingSetInputs")
  println(s"Train exit: \n$trainingSetOutputs")

  def serial(): Unit = Random.setSeed(1)

  def getRandom = 2 * Random.nextDouble() -1
  def getArray(n: Int): Array[Double] = if (n == 1) Array(getRandom) else  Array(getRandom) ++ getArray(n-1)
  val weights: DenseMatrix[Double] = DenseMatrix.create(row, col, getArray(row))
  println(s"First weights \n$weights")

  def think(inputs: DenseMatrix[Int], weights: DenseMatrix[Double]): DenseMatrix[Double] = sigmoid(convert(inputs, Double) * weights)

  def sigmoidDerivative(x: DenseMatrix[Double]): DenseMatrix[Double] = x *:* (-x + 1.0)

  def train(inputs: DenseMatrix[Int], outputs: DenseMatrix[Int]): DenseMatrix[Double] = {

    def trainAux(inputs: DenseMatrix[Int], outputs: DenseMatrix[Int], weights: DenseMatrix[Double], n: Int): DenseMatrix[Double] = {
      n match {
        case 0 => weights
        case _ =>
          val output: DenseMatrix[Double] = think(inputs, weights)
          val error: DenseMatrix[Double] = convert(outputs, Double) - output
          val adjust: DenseMatrix[Double] = convert(inputs, Double).t * (error *:* sigmoidDerivative(output))
          trainAux(inputs, outputs, weights + adjust, n - 1)
      }
    }

    trainAux(inputs, outputs, weights, learning)
  }
  val newW: DenseMatrix[Double] = train(trainingSetInputs, trainingSetOutputs)

  println("Last weights")
  println(newW)

  val newSituation: Array[Int] = Array(1,0,0)
  val result = think(DenseMatrix.create(col,row, newSituation), newW)
  println(s"New situation [${newSituation.mkString(", ")}]: $result ~> ${round(result)}")
}