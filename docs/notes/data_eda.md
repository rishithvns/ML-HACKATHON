# Data & EDA (M1: Rishit)


## 1. What are the 4 files per trip and how do they relate?
Each trip contains four files, and each file provides a different part of the information needed for map matching.
- .track – Contains the actual GPS observations, including longitude, latitude and timestamp. It tells us where the vehicle was recorded over time.
- .nodes – Contains the road network's nodes, where each node represents a geographical location using longitude and latitude.
- .arcs – Connects the nodes and represents the directed road segments. Each arc has a starting node and an ending node.
- .route – Contains the reference route as a sequence of arc IDs in the order in which the vehicle travelled.
All four files work together. The .track file gives us the noisy GPS trajectory, while .nodes and .arcs describe the available road network. The .route file provides the reference route, which we use for training and evaluation, though it is not guaranteed to be error-free.

## 2. Dataset size and quality (partition_summary.csv, data_quality.md)
Our dataset is divided into two partitions: training and student testing.

| Details | Training | Student Test |
|---|---|---|
| Number of trips | 80 | 10 |
| Total GPS points | 208,417 | 19,160 |
| Average points per trip | 2,605 | 1,916 |
| Average duration | 94.67 min | 33.75 min |
| Average road nodes | 24,169 | 9,069 |
| Average road arcs | 48,461 | 18,504 |

Data quality findings:
- We found zero missing values across the raw .track, .nodes, .arcs, and .route files.
- All trip timestamps were already sorted, and no duplicate timestamps were observed.
- Most GPS observations were recorded at approximately 1-second intervals. The average fraction of 1-second intervals was around 93.83% for training and 98.20% for student testing.
- We identified GPS gaps greater than 60 seconds in 22 training trips and 1 student-test trip.
- The largest gap was 153,064 seconds in trip 00000090.
- The typical (median per-trip) GPS-to-reference-route distance was approximately 4.64 metres for training and 4.75 metres for student testing.

Overall, the dataset is reasonably clean, but we identified a few trips with unusual GPS behaviour and some long gaps that need special attention.

## 3. Why project lon/lat to metres?
Initially, GPS coordinates are represented as longitude and latitude in degrees. However, degrees are not suitable for directly calculating distances because one degree of longitude represents different physical distances depending on the latitude.

Our project converts these coordinates into a local metre-based coordinate system using an equirectangular projection centred around each trip.

This makes it easier to calculate distances between GPS points and road segments accurately. It is especially important because our candidate-generation process uses a 50-metre search radius, and our labelling process also uses a 50-metre distance threshold.

In simple terms, we convert GPS coordinates into metres so that all distance calculations are meaningful and consistent.

## 4. How does each GPS point get its label from `.route`?
The .route file contains the correct sequence of road arc IDs, but it does not directly tell us which GPS observation belongs to which arc.

To solve this, we use a monotonic dynamic programming approach in labels.py.

First, we calculate the distance between every GPS observation and the road segments present in the reference route. We also consider the vehicle's heading, because the direction of travel helps distinguish between opposite directions on the same road.

Then, dynamic programming assigns each GPS observation to the most suitable position in the reference route while maintaining chronological order.

For example, if the vehicle travels through arcs A, B and C, the GPS observations should follow that same route order. The algorithm does not allow an observation to go backwards in the route sequence, although it can skip short road segments between GPS observations.

Finally, we calculate the distance between each GPS point and its assigned road segment.

If the distance is greater than 50 metres, the observation is marked as label_valid = False and is excluded from training and point-level evaluation.

This gives us reliable labels without manually assigning a road segment to every GPS point

## 5. Problem trips and how we handled them (33, 54, ...)
During our data-quality analysis, we identified two major problematic trips: 00000033 and 00000054.

These trips showed unusually large distances between their GPS observations and the corresponding reference road segments.

| Trip | Median GPS-route distance | 95th percentile distance | Valid labels |
|---|---|---|---|
| 00000033 | 122.4 m | 142.0 m | 41% |
| 00000054 | 140.0 m | 209.6 m | 11% |

We suspect that these trips have a GPS-to-reference alignment problem because their GPS coordinates appear to be consistently offset from the reference route.

Instead of forcing incorrect labels into our training data, we applied a 50-metre distance threshold. Observations beyond this threshold were marked invalid and excluded from training and point-level scoring.

We did not remove the entire trips. Their valid observations could still be used, while unreliable observations were ignored.

We also identified trips 00000002, 00000066 and 00000043 with smaller labelling issues, which were handled using the same approach.

We also found that 11 reference routes (10 train, 1 student-test) have a few consecutive arcs that are not physically connected (reports/eda/data_quality.md).

This helps prevent poor-quality labels from affecting model training and evaluation.
