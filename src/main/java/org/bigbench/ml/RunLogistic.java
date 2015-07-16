package org.bigbench.ml;

import org.apache.commons.cli2.CommandLine;
import org.apache.commons.cli2.Group;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.builder.ArgumentBuilder;
import org.apache.commons.cli2.builder.DefaultOptionBuilder;
import org.apache.commons.cli2.builder.GroupBuilder;
import org.apache.commons.cli2.commandline.Parser;
import org.apache.commons.cli2.util.HelpFormatter;
import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.classification.LogisticRegressionWithSGD;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import scala.Tuple2;

import java.util.Locale;

public class RunLogistic {

    private static String inputFile = null;
    private static String modelFile = null;
    private static boolean showAuc = true;
    private static boolean showConfusion = true;

    public static void main(String[] args) {

        if (!parseArgs(args)) throw new IllegalArgumentException("Parse arguments failed.");

        SparkConf conf = new SparkConf().setAppName("Logistic Regression with SGD");
        SparkContext sc = new SparkContext(conf);

        JavaRDD<String> data = sc.textFile(inputFile, 1).toJavaRDD();
        JavaRDD<LabeledPoint> test = data.map(new Function<String, LabeledPoint>() {
            public LabeledPoint call(String line) {
                String[] splits = line.split(",");
                double[] features = new double[3];
                try {
                    features[0] = Double.valueOf(splits[1]);
                    features[1] = Double.valueOf(splits[2]);
                    features[2] = Double.valueOf(splits[3]);
                    return new LabeledPoint(
                            Double.valueOf(splits[3]),
                            Vectors.dense(features)
                    );
                } catch (NumberFormatException e) {
                    return null;    // Nothing to do..
                }
            }
        }).filter(new Function<LabeledPoint, Boolean>() {
            public Boolean call(LabeledPoint p) {
                return p != null;
            }
        }).cache();

        final LogisticRegressionModel model = LogisticRegressionModel.load(sc, modelFile);

        JavaRDD<Tuple2<Object, Object>> predicationAndLabels = test.map(
                new Function<LabeledPoint, Tuple2<Object, Object>>() {
                    public Tuple2<Object, Object> call(LabeledPoint p) {
                        return new Tuple2<Object, Object>(model.predict(p.features()), p.label());
                    }
                }
        );

        MulticlassMetrics metrics = new MulticlassMetrics(predicationAndLabels.rdd());

        double acc = -1;
        if (showAuc) {
            acc = metrics.precision();

        }
        Matrix m = null;
        if (showConfusion) {
            m = metrics.confusionMatrix();
        }

        System.out.println("-------------------------------------------------------------");
        if (acc != -1) System.out.printf(Locale.ENGLISH, "AUC = %.2f%n", acc);
        if (m != null) System.out.println("confusion:\n" + m);
        System.out.println("-------------------------------------------------------------");
    }

    private static boolean parseArgs(String[] args) {
        DefaultOptionBuilder builder = new DefaultOptionBuilder();

        Option help = builder.withLongName("help").withDescription("print this list").create();

        Option auc = builder.withLongName("auc").withDescription("print AUC").create();

        Option confusion = builder.withLongName("confusion").withDescription("print confusion matrix").create();

        ArgumentBuilder argumentBuilder = new ArgumentBuilder();
        Option inputFileOption = builder.withLongName("input")
                .withRequired(true)
                .withArgument(argumentBuilder.withName("input").withMaximum(1).create())
                .withDescription("where to get training data")
                .create();

        Option modelFileOption = builder.withLongName("model")
                .withRequired(true)
                .withArgument(argumentBuilder.withName("model").withMaximum(1).create())
                .withDescription("where to get a model")
                .create();

        Group normalArgs = new GroupBuilder()
                .withOption(help)
                .withOption(auc)
                .withOption(confusion)
                .withOption(inputFileOption)
                .withOption(modelFileOption)
                .create();

        Parser parser = new Parser();
        parser.setHelpOption(help);
        parser.setHelpTrigger("--help");
        parser.setGroup(normalArgs);
        parser.setHelpFormatter(new HelpFormatter(" ", "", " ", 130));
        CommandLine cmdLine = parser.parseAndHelp(args);

        if (cmdLine == null) {
            return false;
        }

        inputFile = getStringArgument(cmdLine, inputFileOption);
        modelFile = getStringArgument(cmdLine, modelFileOption);
        showAuc = getBooleanArgument(cmdLine, auc);
        showConfusion = getBooleanArgument(cmdLine, confusion);

        return true;
    }

    private static boolean getBooleanArgument(CommandLine cmdLine, Option option) {
        return cmdLine.hasOption(option);
    }

    private static String getStringArgument(CommandLine cmdLine, Option inputFile) {
        return (String) cmdLine.getValue(inputFile);
    }


}
