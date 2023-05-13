#!/usr/bin/bash

FMOW_CHECKPOINTS="0,1,2
1,2,3
2,3,4
3,4,5
4,5,6
5,6,7
6,7,8
7,8,9
0,2,4
1,3,5
2,4,6
3,5,7
4,6,8
5,7,9
0,3,6
1,4,7
2,5,8
3,6,9
0,4,8
1,5,9
0,5,9"

FMOW_SWA=`seq 0 9 | sed -e 's/$/_swa/'`

YEARBOOK_CHECKPOINTS="1930,1931,1932
1931,1932,1933
1932,1933,1934
1933,1934,1935
1934,1935,1936
1935,1936,1937
1937,1938,1939
1938,1939,1940
1939,1940,1941
1940,1941,1942
1941,1942,1943
1942,1943,1944
1943,1944,1945
1930,1932,1934
1931,1933,1935
1932,1934,1936
1933,1935,1937
1934,1936,1938
1935,1937,1939
1936,1938,1940
1937,1939,1941
1938,1940,1942
1939,1941,1943
1940,1942,1944
1941,1943,1945
1930,1933,1936
1931,1934,1937
1932,1935,1938
1933,1936,1939
1934,1937,1940
1935,1938,1941
1936,1939,1942
1937,1940,1943
1938,1941,1944
1939,1942,1945
1930,1935,1940
1935,1940,1945
1940,1945,1950
1945,1950,1955
1950,1955,1960
1955,1960,1965"

YEARBOOK_SWA=`seq 1930 1965 | sed -e 's/$/_swa/'`

LOG_DIR=" /projects/tir6/strubell/jaredfer/projects/wild-time/results"
EXP_NAME="0504_yearbook_swa_warmstart_csaw_1"
METHOD="swa" # or erm
DATASET="yearbook"

# Outputs png and pdf to ${LOG_DIR}/${EXP_NAME}/loss_contours
for ts in $YEARBOOK_CHECKPOINTS
do
	t0=`echo $ts | cut -d, -f1`
	t1=`echo $ts | cut -d, -f2`
	t2=`echo $ts | cut -d, -f3`
	python main.py --loss_contours --dataset $DATASET --method $METHOD --exp_path $EXP_NAME --contour_timesteps $t2 --contour_models $t0 $t1 $t2 --contour_additional_models ${YEARBOOK_SWA} --contour_margin 0.1 --contour_granularity 50 --contour_increment 0.01;
done