# Learning Social Circles

The data is from [here](https://www.kaggle.com/c/learning-social-circles/data)

Our implementation is based on this [article](https://inventingsituations.net/2014/11/09/kaggle-social-networks-competition/?fbclid=IwAR3q4uz2qKZWpWq0wOMXgffuenFbe8II9O8qCK5R3HuJrCSW7gOEXSqR0rY)

## Data
+ egonets: Each file in this directory contains the ego-network of a single Facebook user, with the form:
```
UserId: Friends
1: 4 6 12 2 208
2: 5 3 17 90 7
```
+ features: Contains features for all users. 
+ Training: Each user.circles file in this directory contains human-labeled circles provided by a user.

## TODO:
+ To generate circles
- [x] Compute an approximation of the exponential adjacency matrix E of the friend graph.
- [x] Apply a generic clustering algorithm using the rows of the E as the feature vectors. 
- [x] Throw away low-density circles.
- [x] Augment the remaining circles by adding in people with more than F friends in the circle, or if they are friends with at least 50% of the people in the circle.
- [x] Include small connected components with at least 5 members and no more than 15 as independent circles.
- [x] Merge circles with more than 75% overlap.
+ To evaluate
- [x] Compare with ground truth

## Usage
Generate the circle for each user.
```
python main.py
```

To evaluate the circles generated in comparison with the ground truth:
```
sh evaluate.sh
```