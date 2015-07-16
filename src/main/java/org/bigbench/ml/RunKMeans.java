package org.bigbench.ml;

import org.apache.commons.cli2.CommandLine;
import org.apache.commons.cli2.Group;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.builder.ArgumentBuilder;
import org.apache.commons.cli2.builder.DefaultOptionBuilder;
import org.apache.commons.cli2.builder.GroupBuilder;
import org.apache.commons.cli2.commandline.Parser;
import org.apache.commons.cli2.util.HelpFormatter;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.clustering.KMeans;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

import java.io.IOException;
import java.util.Collection;
import java.util.List;
import java.util.regex.Pattern;

/**
 * Created with IntelliJ IDEA.
 * User: yantang
 * Date: 7/13/15
 * Time: 3:52 PM
 * To change this template use File | Settings | File Templates.
 */
public class RunKMeans {

    private static String inputFile = null;
    private static int k= 0;
    private static int maxIteration = 0;
    private static int runs=1;
    private static String outputFile=null;
    public static void main(String[] args) {
        if (!parseArgs(args)) throw new IllegalArgumentException("Parse arguments failed.");
        SparkConf sparkConf = new SparkConf().setAppName("JavaKMeans");
        JavaSparkContext sc = new JavaSparkContext(sparkConf);
        JavaRDD<String> lines = sc.textFile(inputFile);

        JavaRDD<Vector> points = lines.map(new ParsePoint());

        KMeansModel model = KMeans.train(points.rdd(), k, maxIteration, runs, KMeans.RANDOM());

        List<Integer> relList=model.predict(points).collect();
        int[] nArr= new int[k];
        for(int i=0;i<relList.size();i++){
            nArr[relList.get(i)]++;
        }

        String content="";
        int cid=0;
        for (Vector center : model.clusterCenters()) {
            content=content+"Cluster "+cid+" " + center+" n="+nArr[cid]+"\r\n";
            cid++;
        }

        createNewHDFSFile(outputFile,content);

        double cost = model.computeCost(points.rdd());
        System.out.println("Cost: " + cost);

        sc.stop();
    }
    public static void createNewHDFSFile(String toCreateFilePath, String content)  {
        FSDataOutputStream os = null;
        FileSystem hdfs = null;
        try {
            Configuration config = new Configuration();
            hdfs = FileSystem.get(config);
            Path inPath=new Path(toCreateFilePath);
            if(hdfs.exists(inPath))
                hdfs.delete(inPath);
            os = hdfs.create(inPath);
            os.write(content.getBytes("UTF-8"));
        }catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }finally{
            try {
                os.close();
                hdfs.close();
            } catch (IOException e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }

        }

    }
    private static class ParsePoint implements Function<String, Vector> {
        private static final Pattern SPACE = Pattern.compile(" ");

        @Override
        public Vector call(String line) {
            String[] tok = SPACE.split(line);
            double[] point = new double[tok.length];
            for (int i = 0; i < tok.length; ++i) {
                point[i] = Double.parseDouble(tok[i]);
            }
            return Vectors.dense(point);
        }
    }
    public static <T> void print(Collection<T> c) {
        for(T t : c) {
            System.out.println(t.toString());
        }
    }
    private static boolean parseArgs(String[] args) {
        DefaultOptionBuilder builder = new DefaultOptionBuilder("-","--",true);

        Option help = builder.withLongName("help").withDescription("print this list").create();
        ArgumentBuilder argumentBuilder = new ArgumentBuilder();
        Option inputFileOption = builder.withLongName("input")
                .withRequired(true)
                .withArgument(argumentBuilder.withName("input").withMaximum(1).create())
                .withDescription("where to get input data")
                .create();
        Option outputFileOption = builder.withLongName("output")
                .withRequired(true)
                .withArgument(argumentBuilder.withName("output").withMaximum(1).create())
                .withDescription("where to write result data")
                .create();
        Option kOption = builder.withLongName("k")
                .withRequired(true)
                .withArgument(argumentBuilder.withName("k").withMaximum(1).create())
                .withDescription("where to get k number")
                .create();

        Option maxIterationOption = builder.withLongName("max_iteration")
                .withRequired(true)
                .withArgument(argumentBuilder.withName("maxIteration").withMaximum(1).create())
                .withDescription("where to get maxIteration number")
                .create();

        Option runsOption = builder.withLongName("runs")
                .withRequired(true)
                .withArgument(argumentBuilder.withName("runs").withMaximum(1).create())
                .withDescription("where to get runs number")
                .create();
        Option hiveOption = builder.withShortName("i")
                .withRequired(false)
                .withArgument(argumentBuilder.withName("i").create())
                .withDescription("hive params")
                .create();
        Option hiveConfOption = builder.withLongName("hiveconf")
                .withRequired(false)
                .withArgument(argumentBuilder.withName("hiveconf").create())
                .withDescription("hiveconf params")
                .create();
        Group normalArgs = new GroupBuilder()
                .withOption(help)
                .withOption(inputFileOption)
                .withOption(outputFileOption)
                .withOption(kOption)
                .withOption(maxIterationOption)
                .withOption(runsOption)
                .withOption(hiveOption)
                .withOption(hiveConfOption)
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
        k = Integer.parseInt(getStringArgument(cmdLine, kOption));
        maxIteration =  Integer.parseInt(getStringArgument(cmdLine, maxIterationOption));
        runs =  Integer.parseInt(getStringArgument(cmdLine, runsOption));
        outputFile= getStringArgument(cmdLine, outputFileOption);
        return true;
    }
    private static String getStringArgument(CommandLine cmdLine, Option inputFile) {
        return (String) cmdLine.getValue(inputFile);
    }

}
