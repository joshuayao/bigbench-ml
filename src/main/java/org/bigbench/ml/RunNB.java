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
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.classification.NaiveBayes;
import org.apache.spark.mllib.classification.NaiveBayesModel;
import org.apache.spark.mllib.feature.HashingTF;
import org.apache.spark.mllib.feature.IDF;
import org.apache.spark.mllib.feature.IDFModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.regression.LabeledPoint;
import scala.Tuple2;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.regex.Pattern;

/**
 * Created with IntelliJ IDEA.
 * User: yantang
 * Date: 7/14/15
 * Time: 4:08 PM
 * To change this template use File | Settings | File Templates.
 */
public class RunNB {
    private static String train_input=null;
    private static String test_input=null;
    private static String model_output=null;
    private static String result_output=null;
    private static SparkConf sparkConf = new SparkConf().setAppName("MLlibNB");
    private static JavaSparkContext sc = new JavaSparkContext(sparkConf);
    private static NaiveBayesModel model;
    private static String[] labelArr={"NEG","NEU","POS"};
    public static void main(String[] args) {
        if (!parseArgs(args)) throw new IllegalArgumentException("Parse arguments failed.");

        // Load the documents
        JavaRDD<String> data = sc.textFile(train_input);
        // Hash all documents
        JavaRDD<LabeledPoint> tupleData = data.map(new ParseDocument(labelArr)).filter(new Function<LabeledPoint, Boolean>() {
            public Boolean call(LabeledPoint p) {
                return p != null;
            }
        }).cache();
        // Create a flat RDD with all vectors
        JavaRDD<Vector> hashedData = tupleData.map(new ParseFeature());
        //  Create a IDFModel out of our flat vector RDD
        IDFModel idfModel = new IDF().fit(hashedData);
        // Create tfidf RDD
        JavaRDD<Vector> idf = idfModel.transform(hashedData);
        // Create Labledpoint RDD
        JavaRDD<LabeledPoint> idfTransformed = idf.zip(tupleData).map(new ParseRDD());
        //train
        model= NaiveBayes.train(idfTransformed.rdd(),1.0);

        model.save(sc.sc(), model_output);

        //process the test_input documents
        JavaRDD<String> data_test = sc.textFile(test_input);
        JavaRDD<LabeledPoint> tupleData_test = data_test.map(new ParseDocument(labelArr)).filter(new Function<LabeledPoint, Boolean>() {
            public Boolean call(LabeledPoint p) {
                return p != null;
            }
        }).cache();
        JavaRDD<Vector> hashedData_test = tupleData_test.map(new ParseFeature());
        IDFModel idfModel_test = new IDF().fit(hashedData_test);
        JavaRDD<Vector> idf_test = idfModel_test.transform(hashedData_test);
        JavaRDD<LabeledPoint> idfTransformed_test = idf_test.zip(tupleData_test).map(new ParseRDD());
        //predict the testing documents
        JavaRDD<Double> prediction = idfTransformed_test.map(new ParseLabelPoint(model));
        JavaRDD<Double> test =   idfTransformed_test.map(new ParseLabel());
        JavaPairRDD<Double, Double> predictionAndLabel = prediction.zip(test);

        //process the classify result
        long correctCn=  predictionAndLabel.filter(new ParseAccuracy()).count();
        long totalCn=  idfTransformed_test.count();

        ResultAnalyzer resultAnalyzer=new ResultAnalyzer(labelArr,correctCn,totalCn,prediction,test);

        createNewHDFSFile(result_output,resultAnalyzer.getResult());


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


    private static class ParseAccuracy implements Function<Tuple2<Double,Double>, Boolean> {

        @Override
        public Boolean call(Tuple2<Double,Double> p) {
            return  p._1().equals(p._2());
        }
    }
    private static class ParseLabel implements Function<LabeledPoint, Double> {

        @Override
        public Double call(LabeledPoint p) {
            return  p.label();
        }
    }
    private static class ParseLabelPoint implements Function<LabeledPoint, Double> {
        private NaiveBayesModel model;
        public ParseLabelPoint(NaiveBayesModel model){
            this.model=model;
        }
        @Override
        public Double call(LabeledPoint p) {
            return  model.predict(p.features());
        }
    }

    private static class ParseRDD implements Function<Tuple2<Vector,LabeledPoint>, LabeledPoint> {

        @Override
        public LabeledPoint call(Tuple2<Vector,LabeledPoint> t) {
            return   new LabeledPoint(t._2().label(), t._1());
        }
    }
    private static class ParseFeature implements Function<LabeledPoint, Vector> {

        @Override
        public Vector call(LabeledPoint v1) {
                          return v1.features();
        }
    }

    private static class ParseDocument implements Function<String, LabeledPoint> {
       private static final Pattern TAB= Pattern.compile("\t");

        private List<String> labelList;
        public ParseDocument(String[] labelArr){
            this.labelList= Arrays.asList(labelArr);
        }
        @Override
        public LabeledPoint call(String line) {


            HashingTF tf = new HashingTF();
            String[] datas = TAB.split(line);
            if(datas.length==3){

                List<String> myList = Arrays.asList(doProcess(datas[2]).split(" "));
                return new LabeledPoint(Double.parseDouble(labelList.indexOf(datas[1]) + ""), tf.transform(myList));
            }
            return null;  //45 items in temp_testing  has empty value in the pr_review_content column. Do nothing for them.

        }
        private String doProcess(String data){
            String result="";
            InputStream inputStream=null;
            try {
                ClassLoader classLoader = getClass().getClassLoader();
                inputStream=classLoader.getResourceAsStream("stopWords.txt");
                BufferedReader stopWordsBR = new BufferedReader(new InputStreamReader(inputStream));
                List<String> stopWordsArray = new ArrayList<String>();
                String stopWordsLine;
                while((stopWordsLine = stopWordsBR.readLine()) != null){
                    if(!stopWordsLine.isEmpty()){
                        stopWordsArray.add(stopWordsLine);
                    }
                }
                String res[] = data.split("[^a-zA-Z]");
                for(int i = 0; i < res.length; i++){
                    if(!res[i].isEmpty() && !stopWordsArray.contains(res[i].toLowerCase())){
                        result += " " + res[i].toLowerCase();
                    }
                }

                return result;
            }catch (FileNotFoundException exception){
                System.out.println("File not exist!");
                exception.printStackTrace();

            }catch (IOException io){
                io.printStackTrace();
            }finally {
                try{
                    inputStream.close();
                }catch (IOException io) {
                    io.getMessage();
                }
            }


           return null;
        }

    }

    private static boolean parseArgs(String[] args) {
        DefaultOptionBuilder builder = new DefaultOptionBuilder("-","--",true);

        Option help = builder.withLongName("help").withDescription("print this list").create();
        ArgumentBuilder argumentBuilder = new ArgumentBuilder();
        Option trinputFileOption = builder.withLongName("train_input")
                .withRequired(true)
                .withArgument(argumentBuilder.withName("train_input").withMaximum(1).create())
                .withDescription("where to get train data")
                .create();
        Option mdoutputFileOption = builder.withLongName("model_output")
                .withRequired(true)
                .withArgument(argumentBuilder.withName("model_output").withMaximum(1).create())
                .withDescription("where to write model data")
                .create();
        Option testinputFileOption = builder.withLongName("test_input")
                .withRequired(true)
                .withArgument(argumentBuilder.withName("test_input").withMaximum(1).create())
                .withDescription("where to get test data")
                .create();
        Option resultoutputFileOption = builder.withLongName("result_output")
                .withRequired(true)
                .withArgument(argumentBuilder.withName("result_output").withMaximum(1).create())
                .withDescription("where to write result data")
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
                .withOption(trinputFileOption)
                .withOption(mdoutputFileOption)
                .withOption(testinputFileOption)
                .withOption(resultoutputFileOption)
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

        train_input = getStringArgument(cmdLine, trinputFileOption);
        model_output =getStringArgument(cmdLine, mdoutputFileOption);
        test_input = getStringArgument(cmdLine, testinputFileOption);
        result_output=getStringArgument(cmdLine,resultoutputFileOption);
        return true;
    }
    private static String getStringArgument(CommandLine cmdLine, Option inputFile) {
        return (String) cmdLine.getValue(inputFile);
    }
}
