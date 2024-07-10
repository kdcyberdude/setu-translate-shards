#!/bin/bash
/usr/local/bin/salad-http-job-queue-worker &
python app.py &
wait -n
exit $?