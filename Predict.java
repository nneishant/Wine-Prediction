package org.njit.np763;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.io.File;

import static org.njit.vb53.Constants.*;

public class Predict {

    private static final Logger logger = Logger.getLogger(Predict.class);

    public static void main(String[] args) {
        configureLoggerLevels();

        SparkSession spark = createSparkSession();

        File testFile = new File(TESTING_DATASET);
        if (testFile.exists()) {
            Predict predictor = new Predict();
            predictor.runLogisticRegression(spark);
        } else {
            System.out.print("TestDataset.csv doesn't exist. Please provide the testFilePath using -v.\n" +
                    "Example: docker run -v [local_testfile_directory:/data] nieldeokar/wine-prediction-mvn:1.0 /TestDataset.csv\n");
        }
    }

    private static void configureLoggerLevels() {
        Logger.getLogger("org").setLevel(Level.ERROR);
        Logger.getLogger("akka").setLevel(Level.ERROR);
        Logger.getLogger("breeze.optimize").setLevel(Level.ERROR);
        Logger.getLogger("com.amazonaws.auth").setLevel(Level.DEBUG);
        Logger.getLogger("com.github").setLevel(Level.ERROR);
    }

    private static SparkSession createSparkSession() {
        return SparkSession.builder()
                .appName(APP_NAME)
                .master("local[*]")
                .config("spark.executor.memory", "2147480000")
                .config("spark.driver.memory", "2147480000")
                .config("spark.testing.memory", "2147480000")
                .getOrCreate();
    }

    public void runLogisticRegression(SparkSession spark) {
        System.out.println("TestingDataSet Metrics\n");

        PipelineModel pipelineModel = PipelineModel.load(MODEL_PATH);
        Dataset<Row> testDf = prepareDataFrame(spark, true, TESTING_DATASET).cache();
        Dataset<Row> predictionDF = pipelineModel.transform(testDf).cache();
        predictionDF.select("features", "label", "prediction").show(5, false);
        printMetrics(predictionDF);
    }

    public Dataset<Row> prepareDataFrame(SparkSession spark, boolean transform, String name) {
        Dataset<Row> validationDf = readCSV(spark, name);
        Dataset<Row> lblFeatureDf = renameColumns(validationDf);
        lblFeatureDf = lblFeatureDf.na().drop().cache();

        VectorAssembler assembler = createVectorAssembler();
        if (transform) {
            lblFeatureDf = assembler.transform(lblFeatureDf).select("label", "features");
        }

        return lblFeatureDf;
    }

    private Dataset<Row> readCSV(SparkSession spark, String name) {
        return spark.read().format("csv")
                .option("header", "true")
                .option("multiline", true)
                .option("sep", ";")
                .option("quote", "\"")
                .option("dateFormat", "M/d/y")
                .option("inferSchema", true)
                .load(name);
    }

    private Dataset<Row> renameColumns(Dataset<Row> validationDf) {
        return validationDf.withColumnRenamed("quality", "label")
                .select("label", FEATURE_COLUMNS);
    }

    private VectorAssembler createVectorAssembler() {
        return new VectorAssembler().setInputCols(FEATURE_COLUMNS).setOutputCol("features");
    }

    public void printMetrics(Dataset<Row> predictions) {
        System.out.println();
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator();
        evaluator.setMetricName("accuracy");
        System.out.println("The accuracy of the model is " + evaluator.evaluate(predictions));

        evaluator.setMetricName("f1");
        double f1 = evaluator.evaluate(predictions);

        System.out.println("F1: " + f1);
    }
}
