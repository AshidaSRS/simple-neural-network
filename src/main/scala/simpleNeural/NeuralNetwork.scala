package simpleNeural

import breeze.linalg._
import breeze.numerics._

case class NeuralNetwork(layer1: NeuronLayer, layer2: NeuronLayer)

object NeuralNetwork {

  val learning = 10000

  def apply(layer1: NeuronLayer, layer2: NeuronLayer): NeuralNetwork =
    new NeuralNetwork(layer1, layer2)

  def think(nn: NeuralNetwork, inputs: DenseMatrix[Int]): Think2 = {
    val outputL1 = sigmoid(convert(inputs, Double) * nn.layer1.synapse)
    val outputL2 = sigmoid(outputL1 * nn.layer2.synapse)
    Think2(outputL1, outputL2)
  }

  def sigmoidDerivative(x: DenseMatrix[Double]): DenseMatrix[Double] = x *:* (-x + 1.0)

  def train(nn: NeuralNetwork, inputs: DenseMatrix[Int], outputs: DenseMatrix[Int], n: Int): NeuralNetwork = {

    def trainAux(nn: NeuralNetwork, tInputs: DenseMatrix[Int], tOutputs: DenseMatrix[Int], n: Int): NeuralNetwork = {
      n match {
        case 0 => nn
        case _ =>
          val outputs: Think2 = think(nn, tInputs)

          val errorsL2 = convert(tOutputs, Double) - outputs.neuron2
          val errorsL2Delta = errorsL2 *:* sigmoidDerivative(outputs.neuron2)

          val errorsL1 = errorsL2Delta * nn.layer2.synapse.t
          val errorsL1Delta = errorsL1 *:* sigmoidDerivative(outputs.neuron1)

          val adjustmentL1 = convert(tInputs, Double).t * errorsL1Delta
          val adjustmentL2 = outputs.neuron1.t * errorsL2Delta

          val newNeural = nn.copy(
            layer1 = nn.layer1.copy(synapse = nn.layer1.synapse + adjustmentL1),
            layer2 = nn.layer2.copy(synapse = nn.layer2.synapse + adjustmentL2)
          )
          trainAux(newNeural, inputs, tOutputs, n - 1)
      }
    }
    trainAux(nn, inputs, outputs, n)
  }
}
