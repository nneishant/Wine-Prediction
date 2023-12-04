package org.njit.np763;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.classification.LogisticRegressionTrainingSummary;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.io.File;
import java.io.IOException;

import static org.njit.np763.Constants.*;

public class ModelTrainer {

    private static final Logger logger = Logger.getLogger(ModelTrainer.class);

    public static void main(String[] args) {
        configureLoggerLevels();
        SparkSession spark = createSparkSession();

        File trainingFile = new File(TRAINING_DATASET);
        if (trainingFile.exists()) {
            ModelTrainer trainer = new ModelTrainer();
            trainer.trainModel(spark);
        } else {
            System.out.print("TrainingDataset.csv doesn't exist");
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

    public void trainModel(SparkSession spark) {
        System.out.println();
        Dataset<Row> labeledFeatureDf = prepareDataFrame(spark, true, TRAINING_DATASET).cache();
        LogisticRegression logReg = new LogisticRegression().setMaxIter(100).setRegParam(0.0);

        Pipeline pipeline = new Pipeline().setStages(new PipelineStage[]{logReg});
        PipelineModel model = pipeline.fit(labeledFeatureDf);

        LogisticRegressionModel lrModel = (LogisticRegressionModel) (model.stages()[0]);
        LogisticRegressionTrainingSummary trainingSummary = lrModel.summary();
        double accuracy = trainingSummary.accuracy();
        double fMeasure = trainingSummary.weightedFMeasure();

        System.out.println();
        System.out.println("Training DataSet Metrics ");
        System.out.println("Accuracy: " + accuracy);
        System.out.println("F-measure: " + fMeasure);

        Dataset<Row> validationDf = prepareDataFrame(spark, true, VALIDATION_DATASET).cache();
        Dataset<Row> results = model.transform(validationDf);

        System.out.println("\n Validation Training Set Metrics");
        results.select("features", "label", "prediction").show(5, false);
        printMetrics(results);

        try {
            model.write().overwrite().save(MODEL_PATH);
        } catch (IOException e) {
            logger.error("Error saving the model", e);
        }
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

    public Dataset<Row> prepareDataFrame(SparkSession spark, boolean transform, String name) {
        Dataset<Row> validationDf = spark.read().format("csv").option("header", "true")
                .option("multiline", true).option("sep", ";").option("quote", "\"")
                .option("dateFormat", "M/d/y").option("inferSchema", true).load(name);

        validationDf = renameColumns(validationDf);
        validationDf.show(5);

        Dataset<Row> labeledFeatureDf = validationDf.select("label", FEATURE_COLUMNS);
        labeledFeatureDf = labeledFeatureDf.na().drop().cache();

        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(FEATURE_COLUMNS)
                .setOutputCol("features");

        if (transform) {
            labeledFeatureDf = assembler.transform(labeledFeatureDf).select("label", "features");
        }

        return labeledFeatureDf;
    }

    private Dataset<Row> renameColumns(Dataset<
