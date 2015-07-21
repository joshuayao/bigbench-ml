package org.bigbench.ml;

import org.apache.commons.lang3.StringUtils;
import org.apache.spark.api.java.JavaRDD;

import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.List;

/**
 * Created by yantang on 7/20/2015.
 */
public class ResultAnalyzer {
    private JavaRDD<Double> predictionLabels, testLabels;
    private long[][] confusionMatrix;
    private long[] labelCntArr;
    private long correctCnt, totalCnt;
    private int labelCnt;
    private String[] labelArr;

    public ResultAnalyzer(String[] labelArr, long correctCnt, long totalCnt, JavaRDD<Double> predictionLabels, JavaRDD<Double> testLabels) {
        this.labelArr = labelArr;
        this.correctCnt = correctCnt;
        this.totalCnt = totalCnt;
        this.predictionLabels = predictionLabels;
        this.testLabels = testLabels;
        labelCnt = labelArr.length;
        confusionMatrix = new long[labelCnt][labelCnt];
        labelCntArr = new long[labelCnt];



    }

    public String getResult() {
        StringBuilder returnString = new StringBuilder();

        returnString.append('\n');
        returnString.append("=======================================================\n");
        returnString.append("Summary\n");
        returnString.append("-------------------------------------------------------\n");
        long incorrectCnt = totalCnt - correctCnt;
        double percentageCorrect = (double) 100 * correctCnt / totalCnt;
        double percentageIncorrect = (double) 100 * incorrectCnt / totalCnt;
        NumberFormat decimalFormatter = new DecimalFormat("0.####");

        returnString.append(StringUtils.rightPad("Correctly Classified Instances", 40)).append(": ").append(
                StringUtils.leftPad(Long.toString(correctCnt), 10)).append('\t').append(
                StringUtils.leftPad(decimalFormatter.format(percentageCorrect), 10)).append("%\n");
        returnString.append(StringUtils.rightPad("Incorrectly Classified Instances", 40)).append(": ").append(
                StringUtils.leftPad(Long.toString(incorrectCnt), 10)).append('\t').append(
                StringUtils.leftPad(decimalFormatter.format(percentageIncorrect), 10)).append("%\n");
        returnString.append(StringUtils.rightPad("Total Classified Instances", 40)).append(": ").append(
                StringUtils.leftPad(Long.toString(totalCnt), 10)).append('\n');
        returnString.append('\n');
        buildConfusionMatrix();
        returnString.append(getConfusionMatrixStr());

        return returnString.toString();
    }

    private void buildConfusionMatrix() {
        List<Double> testLabelList = testLabels.collect();
        List<Double> predictionLabelList = predictionLabels.collect();
        int testTemp, predicTemp;
        for (int k = 0; k < testLabelList.size(); k++) {
            testTemp = testLabelList.get(k).intValue();
            predicTemp = predictionLabelList.get(k).intValue();
            labelCntArr[testTemp]++;
            if (testTemp == predicTemp)
                confusionMatrix[testTemp][testTemp]++;
            else {
                confusionMatrix[testTemp][predicTemp]++;
            }
        }
    }

    private String getSmallLabel(int i) {
        int val = i;
        StringBuilder returnString = new StringBuilder();
        do {
            int n = val % 26;
            returnString.insert(0, (char) ('a' + n));
            val /= 26;
        } while (val > 0);
        return returnString.toString();
    }

    private String getConfusionMatrixStr() {
        StringBuilder returnString = new StringBuilder(200);
        returnString.append("=======================================================").append('\n');
        returnString.append("Confusion Matrix\n");
        returnString.append("-------------------------------------------------------").append('\n');
        for(int i=0;i<labelCnt;i++)
            returnString.append(StringUtils.rightPad(getSmallLabel(i), 5)).append('\t');
        returnString.append("<--Classified as").append('\n');
        for (int i = 0; i < labelCnt; i++) {
            for (int j = 0; j < labelCnt; j++)
                returnString.append(StringUtils.rightPad(Long.toString(confusionMatrix[i][j]), 5)).append('\t');
            returnString.append(" |  ").append(StringUtils.rightPad(String.valueOf(labelCntArr[i]), 6)).append('\t')
                    .append(StringUtils.rightPad(getSmallLabel(i), 5))
                    .append(" = ").append(labelArr[i]).append('\n');

            returnString.append('\n');

        }
        return returnString.toString();
    }
}
