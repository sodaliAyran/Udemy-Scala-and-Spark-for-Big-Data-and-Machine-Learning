import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.regression.LinearRegression

import org.apache.log4j._

Logger.getLogger("org").setLevel(Level.ERROR)

val spark = SparkSession.builder().getOrCreate()

val data = spark.read.option("header", "true").option("inferSchema", "true").format("csv").load("data/Clean-USA-Housing.csv")

data.printSchema

/*
val colnames = data.columns
val firstrow = data.head(1)(0)
println("\n")
println("Example Data Row")
for(ind <- Range(1,colnames.length)){
  println(colnames(ind))
  println(firstrow(ind))
  println("\n")
}
*/

// ("Label", "features")
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors

val df = (data.select(data("Price").as("label"),
$"Avg Area Income", $"Avg Area House Age", $"Avg Area Number of Rooms", $"Avg Area Number of Bedrooms", $"Area Population"))


val assembler = (new VectorAssembler()
                .setInputCols(Array("Avg Area Income", "Avg Area House Age",
                  "Avg Area Number of Rooms", "Avg Area Number of Bedrooms",
                  "Area Population")).setOutputCol("features"))


val output = assembler.transform(df).select($"label", $"features") // Important!

val lr = new LinearRegression()

// No train test split
val lrModel = lr.fit(output)

val trainingSummary = lrModel.summary

trainingSummary.residuals.show() // difference betwen predicted and actual values
