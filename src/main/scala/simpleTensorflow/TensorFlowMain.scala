package simpleTensorflow

import org.platanios.tensorflow.api._

object TensorFlowMain {

  val inputs: Output = tf.placeholder(FLOAT32, Shape(-1, 10))
  val outputs: Output = tf.placeholder(FLOAT32, Shape(-1, 10))
  val predictions: Output = tf.createWith(nameScope = "Linear") {
    val weights = tf.variable("weights", FLOAT32, Shape(10, 1), tf.ZerosInitializer)
    tf.matmul(inputs, weights)
  }
  val loss: Output = tf.sum(tf.square(predictions - outputs))
  val optimizer = tf.train.AdaGrad(1.0)
  val trainOp: Op = optimizer.minimize(loss)

  def run() = {
    println(trainOp.name)
    println(trainOp.outputs)
  }
}
