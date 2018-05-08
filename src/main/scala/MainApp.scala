import breeze.linalg._

object MainApp extends App {

  val set1 = DenseVector(0,0,1)
  val set2 = DenseVector(0,1,1)
  val set3 = DenseVector(1,0,1)
  val set4 = DenseVector(0,1,0)
  val set5 = DenseVector(1,0,0)
  val set6 = DenseVector(1,1,1)
  val set7 = DenseVector(0,0,0)

  val outSet = DenseVector(0,1,1,1,1,0,0)

  val trainingSetInputs = DenseMatrix(set1, set2, set3, set4, set5, set6, set7)
  val trainingSetOutputs = DenseMatrix(outSet).t

  val layer1: NeuronLayer = NeuronLayer(4, 3)
  val layer2: NeuronLayer = NeuronLayer(1, 4)

  val neuralNetwork: NeuralNetwork = NeuralNetwork(layer1, layer2)

  println("Random start weights")
  println(s"Layer 1: \n${neuralNetwork.layer1.weights.data.mkString("\n")}")
  println(s"Layer 2: \n${neuralNetwork.layer2.weights.data.mkString("\n")}")

  println("Training Inputs")
  println(trainingSetInputs)

  println("Trainig Outputs")
  println(trainingSetOutputs)

  val newNeuralN: NeuralNetwork =
    NeuralNetwork.train(neuralNetwork, trainingSetInputs, trainingSetOutputs, 60000)

  println("New start weights")
  println(s"Layer 1: \n${newNeuralN.layer1.weights.data.mkString("\n")}")
  println(s"Layer 2: \n${newNeuralN.layer2.weights.data.mkString("\n")}")

  val newSituation = Array(1,1,0)
  println(s"New situation [${newSituation.mkString(", ")}] -> ?")
  val newThink: Think2 = NeuralNetwork.think(newNeuralN, DenseMatrix.create(1, 3, newSituation))
  println(s"Hidden state: \n${newThink.neuron1}")
  println(s"Result: \n${newThink.neuron2}")
}