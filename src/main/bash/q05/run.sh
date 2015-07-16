#!/usr/bin/env bash

#"INTEL CONFIDENTIAL"
#Copyright 2015  Intel Corporation All Rights Reserved. 
#
#The source code contained or described herein and all documents related to the source code ("Material") are owned by Intel Corporation or its suppliers or licensors. Title to the Material remains with Intel Corporation or its suppliers and licensors. The Material contains trade secrets and proprietary and confidential information of Intel or its suppliers and licensors. The Material is protected by worldwide copyright and trade secret laws and treaty provisions. No part of the Material may be used, copied, reproduced, modified, published, uploaded, posted, transmitted, distributed, or disclosed in any way without Intel's prior express written permission.
#
#No license under any patent, copyright, trade secret or other intellectual property right is granted to or conferred upon you by disclosure or delivery of the Materials, either expressly, by implication, inducement, estoppel or otherwise. Any license under such intellectual property rights must be express and approved by Intel in writing.

HDFS_RESULT_FILE="${RESULT_DIR}/logRegResult.txt"

query_run_main_method () {
	QUERY_SCRIPT="$QUERY_SQL_DIR/$QUERY_NAME.sql"
	if [ ! -r "$QUERY_SCRIPT" ]
	then
		echo "SQL file $QUERY_SCRIPT can not be read."
		exit 1
	fi

	#EXECUTION Plan:
	#step 1.  hive q05.sql		:	Run hive querys to extract the input data
	#step 2.  mahout TrainLogistic	:	Train logistic regression model
	#step 3.  mahout calc log_reg 	:	Calculating logistic regression for input data
	#step 4.  mahout dump > hdfs/res:	Converting result and copy result do hdfs query result folder
	#step 5.  hive && hdfs 		:	cleanup.sql && hadoop fs rm MH

	RETURN_CODE=0
	if [[ -z "$DEBUG_QUERY_PART" || $DEBUG_QUERY_PART -eq 1 ]] ; then
		echo "========================="
		echo "$QUERY_NAME Step 1/3: Executing hive queries"
		echo "tmp output: ${TEMP_DIR}"
		echo "========================="
		# Write input for k-means into ctable
		runCmdWithErrorCheck runEngineCmd -f "$QUERY_SCRIPT"
		RETURN_CODE=$?
		if [[ $RETURN_CODE -ne 0 ]] ;  then return $RETURN_CODE; fi
	fi
	
	if [[ -z "$DEBUG_QUERY_PART" || $DEBUG_QUERY_PART -eq 2 ]] ; then
	
		echo "========================="
		echo "$QUERY_NAME Step 2/3: log regression"
		echo "========================="

		TMP_LOG_REG_IN_FILE="`mktemp`"
		TMP_LOG_REG_MODEL_FILE="`mktemp`"

		echo "-------------------------"
		echo "$QUERY_NAME Step 2/3 Part 1: Copy hive result to local csv file"
		echo "tmp output: ${TMP_LOG_REG_IN_FILE}"
		echo "-------------------------"

		echo "streaming result from hive ..."
		#write header
		runCmdWithErrorCheck echo '"c_customer_sk","college_education","male","label"' > "${TMP_LOG_REG_IN_FILE}"
		# append hive result
		runCmdWithErrorCheck hadoop fs -cat "${TEMP_DIR}"/* >> "${TMP_LOG_REG_IN_FILE}"
		echo "streaming result from hive ... done"
		echo "sample:"
		echo "size: " `du -bh "${TMP_LOG_REG_IN_FILE}"`
		echo "------"
		head "${TMP_LOG_REG_IN_FILE}"
		echo "..." 
		echo "-------------------------"

		echo "$QUERY_NAME Step 2/3 Part 2: Train logistic model"
		TMP_LOG_REG_MODEL_FILE=${TMP_LOG_REG_MODEL_FILE/./_}
                echo "Command " /usr/lib/spark/bin/spark-submit -v --master local[*] --class org.bigbench.ml.TrainLogistic file://${BIG_BENCH_QUERIES_DIR}/Resources/bigbench-ml.jar  --input "file://$TMP_LOG_REG_IN_FILE" --output "file://$TMP_LOG_REG_MODEL_FILE"  --passes 20 --rate 1 --lambda 0.1
		echo "tmp output: ${TMP_LOG_REG_MODEL_FILE}"
		echo "-------------------------"
	
		runCmdWithErrorCheck /usr/lib/spark/bin/spark-submit -v --master local[*] --class org.bigbench.ml.TrainLogistic file://${BIG_BENCH_QUERIES_DIR}/Resources/bigbench-ml.jar --input "file://$TMP_LOG_REG_IN_FILE" --output "file://$TMP_LOG_REG_MODEL_FILE"  --passes 20 --rate 1 --lambda 0.1
		RETURN_CODE=$?
		if [[ $RETURN_CODE -ne 0 ]] ;  then return $RETURN_CODE; fi
		
		echo "-------------------------"
		echo "$QUERY_NAME Step 2/3 Part 3: Calculating Logistic Regression"
		echo "Command: " /usr/lib/spark/bin/spark-submit -v --master local[*] --class org.bigbench.ml.RunLogistic file://${BIG_BENCH_QUERIES_DIR}/Resources/bigbench-ml.jar --input "file://$TMP_LOG_REG_IN_FILE" --model "file://$TMP_LOG_REG_MODEL_FILE" --auc --confusion
		echo "output: hdfs://"$HDFS_RESULT_FILE
		echo "-------------------------"

		runCmdWithErrorCheck /usr/lib/spark/bin/spark-submit -v --master local[*] --class org.bigbench.ml.RunLogistic file://${BIG_BENCH_QUERIES_DIR}/Resources/bigbench-ml.jar --input "file://$TMP_LOG_REG_IN_FILE" --model "file://$TMP_LOG_REG_MODEL_FILE" --auc --confusion 2> /dev/null | grep -A 3 "AUC =" | hadoop fs -copyFromLocal -f - "$HDFS_RESULT_FILE"
		RETURN_CODE=$?
		if [[ $RETURN_CODE -ne 0 ]] ;  then return $RETURN_CODE; fi
		
		echo "-------------------------"
		echo "$QUERY_NAME Step 2/3 Part 4: Cleanup tmp files"
		echo "-------------------------"
		rm -f "$TMP_LOG_REG_IN_FILE"
		rm -f "$TMP_LOG_REG_MODEL_FILE"
	fi

	if [[ -z "$DEBUG_QUERY_PART" || $DEBUG_QUERY_PART -eq 3 ]] ; then
		echo "========================="
		echo "$QUERY_NAME Step 3/3: Clean up"
		echo "========================="
	
		runCmdWithErrorCheck runEngineCmd -f "${QUERY_DIR}/cleanup.sql"
		RETURN_CODE=$?
		if [[ $RETURN_CODE -ne 0 ]] ;  then return $RETURN_CODE; fi
		
		runCmdWithErrorCheck hadoop fs -rm -r -f "$TEMP_DIR"
		RETURN_CODE=$?
		if [[ $RETURN_CODE -ne 0 ]] ;  then return $RETURN_CODE; fi
		
	fi
}

query_run_clean_method () {
	runCmdWithErrorCheck runEngineCmd -e "DROP TABLE IF EXISTS $TEMP_TABLE; DROP TABLE IF EXISTS $RESULT_TABLE;"
	return $?
}

query_run_validate_method () {
	VALIDATION_TEMP_FILE="`mktemp -u`"
	runCmdWithErrorCheck hadoop fs -copyToLocal "$HDFS_RESULT_FILE" "$VALIDATION_TEMP_FILE"
	if [ `wc -l < "$VALIDATION_TEMP_FILE"` -ge 1 ]
	then
		echo "Validation passed: Query returned results"
	else
		echo "Validation failed: Query did not return results"
	fi
	rm -rf "$VALIDATION_TEMP_FILE"
}
