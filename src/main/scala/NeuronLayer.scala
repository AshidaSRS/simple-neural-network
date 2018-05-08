import breeze.linalg.DenseMatrix
import breeze.stats.distributions.Rand

import scala.util.Random

case class NeuronLayer(total: Int, neuronInputs: Int, synapse: DenseMatrix[Double])

object NeuronLayer {

  def serial(): Unit = Random.setSeed(1)

  def apply(total: Int, neuronInputs: Int): NeuronLayer = {
    def weights = DenseMatrix.rand(neuronInputs, total, Rand.uniform.condition(p => p <= 1 && p >= -1))
    new NeuronLayer(total, neuronInputs, weights)
  }
}