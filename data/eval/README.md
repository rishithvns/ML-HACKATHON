# Evaluation trips (Laptop 1 only)

Copy the 10 reserved evaluator trips here, one folder per trip, same layout as
the raw data:

```
data/eval/<trip_id>/<trip_id>.track
data/eval/<trip_id>/<trip_id>.nodes
data/eval/<trip_id>/<trip_id>.arcs
data/eval/<trip_id>/<trip_id>.route   # server-side scoring only
```

The API lists every folder here in `GET /trips`. The `.route` file is read
only by `pathsnap/evaluation.py` after the prediction is made; it is never sent
to the UI. To rehearse with the student-test trips instead, start the API with
`PATHSNAP_EVAL_DIR=data/raw/student_test`.
