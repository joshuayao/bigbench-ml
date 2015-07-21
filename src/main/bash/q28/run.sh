#!/usr/bin/env bash

#"INTEL CONFIDENTIAL"
#Copyright 2015  Intel Corporation All Rights Reserved. 
#
#The source code contained or described herein and all documents related to the source code ("Material") are owned by Intel Corporation or its suppliers or licensors. Title to the Material remains with Intel Corporation or its suppliers and licensors. The Material contains trade secrets and proprietary and confidential information of Intel or its suppliers and licensors. The Material is protected by worldwide copyright and trade secret laws and treaty provisions. No part of the Material may be used, copied, reproduced, modified, published, uploaded, posted, transmitted, distributed, or disclosed in any way without Intel's prior express written permission.
#
#No license under any patent, copyright, trade secret or other intellectual property right is granted to or conferred upon you by disclosure or delivery of the Materials, either expressly, by implication, inducement, estoppel or otherwise. Any license under such intellectual property rights must be express and approved by Intel in writing.

TEMP_TABLE1="${TEMP_TABLE}_training"
TEMP_DIR1="$TEMP_DIR/$TEMP_TABLE1"
TEMP_TABLE2="${TEMP_TABLE}_testing"
TEMP_DIR2="$TEMP_DIR/$TEMP_TABLE2"

BINARY_PARAMS="$BINARY_PARAMS --hiveconf TEMP_TABLE1=$TEMP_TABLE1 --hiveconf TEMP_DIR1=$TEMP_DIR1 --hiveconf TEMP_TABLE2=$TEMP_TABLE2 --hiveconf TEMP_DIR2=$TEMP_DIR2"

HDFS_RESULT_FILE="${RESULT_DIR}/MLlibClassifierResult.txt"

query_run_main_method () {
	QUERY_SCRIPT="$QUERY_SQL_DIR/$QUERY_NAME.sql"
	if [ ! -r "$QUERY_SCRIPT" ]
	then
		echo "SQL file $QUERY_SCRIPT can not be read."
		exit 1
	fi

	if [[ -z "$DEBUG_QUERY_PART" || $DEBUG_QUERY_PART -eq 1 ]] ; then
		echo "========================="
		echo "$QUERY_NAME step 1/3: Executing hive queries"
		echo "tmp result in" ${TEMP_DIR1}
		echo "tmp result in" ${TEMP_DIR2}
		echo "========================="

		# Write input for k-means into temp tables
		runCmdWithErrorCheck runEngineCmd -f "$QUERY_SCRIPT"
		RETURN_CODE=$?
		if [[ $RETURN_CODE -ne 0 ]] ;  then return $RETURN_CODE; fi
	fi



	if [[ -z "$DEBUG_QUERY_PART" || $DEBUG_QUERY_PART -eq 2 ]] ; then
		echo "========================="
		echo "$QUERY_NAME step 2/3: Running MLlib Naive Bayes"
		echo "OUT: $HDFS_RESULT_FILE"
		echo "========================="
		runCmdWithErrorCheck runEngineCmd --class  org.bigbench.ml.RunNB --train_input $TEMP_DIR1 --model_output $TEMP_DIR --test_input $TEMP_DIR2 --result_output $HDFS_RESULT_FILE
		RETURN_CODE=$?
		if [[ $RETURN_CODE -ne 0 ]] ;  then return $RETURN_CODE; fi
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

	echo "========================="
	echo "to display : hadoop fs -cat $HDFS_RESULT_FILE"
	echo "========================="
}

query_run_clean_method () {
	runCmdWithErrorCheck runEngineCmd -e "DROP TABLE IF EXISTS $TEMP_TABLE1; DROP TABLE IF EXISTS $TEMP_TABLE2; DROP TABLE IF EXISTS $RESULT_TABLE;"
	runCmdWithErrorCheck hadoop fs -rm -r -f "$HDFS_RESULT_FILE"
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
