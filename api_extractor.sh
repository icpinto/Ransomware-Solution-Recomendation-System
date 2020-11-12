#!/bin/bash
file=$1
task_id=$2
proc="$(jq  '.behavior.processes[].process_path' $file | wc -l )"
proc=$((proc-1))
for j in $(seq 0 $proc);
	 do
		
		 seq="$(jq --argjson n "$j"  '.behavior.processes[$n].calls[].api' $file |  sed 's/[^a-zA-Z0-9]//g')" 
		 #seq="$(jq --argjson n "$j" '.behavior.processes[$n].calls[].api' 5.json |  sed 's/[^a-zA-Z0-9]//g')"
		echo $seq >> ${task_id}.txt
		tr '\n' ' ' < ${task_id}.txt > ${task_id}.csv
		echo "" >> ${task_id}.csv
	 done
sed -i.bak 1i"API seq" ${task_id}.csv

