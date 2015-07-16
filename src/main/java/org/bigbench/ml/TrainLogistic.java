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
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.optimization.L1Updater;
import org.apache.spark.mllib.regression.LabeledPoint;

import java.util.Arrays;

public class TrainLogistic {

    private static String inputFile = null;
    private static String outputFile = null;
    private static int passes = 1;
    private static LogisticRegressionWithSGD lrs = new LogisticRegressionWithSGD();

    public static void main(String[] args) {
        if (!parseArgs(args)) throw new IllegalArgumentException("Parse arguments failed.");

        SparkConf conf = new SparkConf().setAppName("Logistic Regression with SGD");
        SparkContext sc = new SparkContext(conf);

        JavaRDD<String> data = sc.textFile(inputFile, 1).toJavaRDD();
        JavaRDD<LabeledPoint> training = data.map(new Function<String, LabeledPoint>() {
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

        LogisticRegressionModel model = lrs.run(training.rdd());
        model.save(sc, outputFile);
        sc.stop();
    }

    public static boolean parseArgs(String[] args) {
        DefaultOptionBuilder builder = new DefaultOptionBuilder();

        Option help = builder.withLongName("help").withDescription("print this list").create();

        ArgumentBuilder argumentBuilder = new ArgumentBuilder();
        Option inputFile = builder.withLongName("input")
                .withRequired(true)
                .withArgument(argumentBuilder.withName("input").withMaximum(1).create())
                .withDescription("where to get training data")
                .create();

        Option outputFile = builder.withLongName("output")
                .withRequired(true)
                .withArgument(argumentBuilder.withName("output").withMaximum(1).create())
                .withDescription("where to get training data")
                .create();

        Option passes = builder.withLongName("passes")
                .withArgument(
                        argumentBuilder.withName("passes")
                                .withDefault("2")
                                .withMaximum(1).create())
                .withDescription("the number of times to pass over the input data")
                .create();

        Option lambda = builder.withLongName("lambda")
                .withArgument(argumentBuilder.withName("lambda").withDefault("1e-4").withMaximum(1).create())
                .withDescription("the amount of coefficient decay to use")
                .create();

        Option rate = builder.withLongName("rate")
                .withArgument(argumentBuilder.withName("learningRate").withDefault("1e-3").withMaximum(1).create())
                .withDescription("the learning rate")
                .create();

        Group normalArgs = new GroupBuilder()
                .withOption(help)
                .withOption(inputFile)
                .withOption(outputFile)
                .withOption(passes)
                .withOption(lambda)
                .withOption(rate)
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

        TrainLogistic.inputFile = getStringArgument(cmdLine, inputFile);
        TrainLogistic.outputFile = getStringArgument(cmdLine, outputFile);
        TrainLogistic.passes = getIntegerArgument(cmdLine, passes);

        lrs.optimizer()
                .setStepSize(getDoubleArgument(cmdLine, rate))
                .setUpdater(new L1Updater())
                .setRegParam(getDoubleArgument(cmdLine, lambda))
                .setNumIterations(TrainLogistic.passes)
                .setMiniBatchFraction(1.0);
        lrs.setIntercept(true);

        return true;

    }

    private static String getStringArgument(CommandLine cmdLine, Option inputFile) {
        return (String) cmdLine.getValue(inputFile);
    }

    private static double getDoubleArgument(CommandLine cmdLine, Option op) {
        return Double.parseDouble((String) cmdLine.getValue(op));
    }

    private static int getIntegerArgument(CommandLine cmdLine, Option features) {
        return Integer.parseInt((String) cmdLine.getValue(features));
    }

}
