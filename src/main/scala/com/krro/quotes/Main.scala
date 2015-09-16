import org.apache.spark.mllib.regression.{ RidgeRegressionWithSGD, LabeledPoint }
import org.apache.spark.{ SparkConf, SparkContext }
import org.apache.spark.mllib.linalg.Vectors
import scala.io.Source

object Main extends App {
  val sparkConfig = new SparkConf().setAppName("quotes").setMaster("local")
  val sparkContext = new SparkContext(sparkConfig)

  val quotesFileLines = Source.fromFile("...your...path...").getLines.toList

  val prices = quotesFileLines.map { _.split(",").toList(5).toDouble }

  val growths = prices.drop(1).zip(prices.dropRight(1)).map {
    case (current, previous) => 100.0 * (current - previous) / previous
  }

  val probesNumber = 20
  val labeledPoints = for(i <- probesNumber until growths.size) yield {
    LabeledPoint(growths(i), Vectors.dense(growths.slice(i - probesNumber, i).toArray))
  }

  val labeledPointsRDD = sparkContext.parallelize(labeledPoints)

  val Array(trainingData, testData) = labeledPointsRDD.randomSplit(Array(0.7, 0.3))

  val numIterations = 1000
  val stepSize = 0.005
  val regularizationParam = 0.01

  val model = RidgeRegressionWithSGD.train(trainingData, numIterations, stepSize, regularizationParam)

  val scoreAndLabels = testData.map { point =>
    val score = model.predict(point.features)
    (point.label, score)
  }

  val meanSquaredError = scoreAndLabels.map { case(l, s) => math.pow((l - s), 2) }.mean
  println("mean squared error = " + meanSquaredError)
}