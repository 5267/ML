## Machine Learning Algorithms
Broadly, there are 3 types of Machine Learning Algorithms.

1. Supervised Learning

**How it works**: This algorithm consist of a target / outcome variable (or dependent variable) which is to be predicted from a given set of predictors (independent variables). Using these set of variables, we generate a function that map inputs to desired outputs. The training process continues until the model achieves a desired level of accuracy on the training data. Examples of Supervised Learning: Regression, Decision Tree, Random Forest, KNN, Logistic Regression etc.

2. Unsupervised Learning

**How it works**: In this algorithm, we do not have any target or outcome variable to predict / estimate.  It is used for clustering population in different groups, which is widely used for segmenting customers in different groups for specific intervention. Examples of Unsupervised Learning: Apriori algorithm, K-means.

3. Reinforcement Learning:

**How it works**:  Using this algorithm, the machine is trained to make specific decisions. It works this way: the machine is exposed to an environment where it trains itself continually using trial and error. This machine learns from past experience and tries to capture the best possible knowledge to make accurate business decisions. Example of Reinforcement Learning: Markov Decision Process

### List of Common Machine Learning Algorithms
Here is the list of commonly used machine learning algorithms. These algorithms can be applied to almost any data problem

1. Linear Regression
2. Logistic Regression
3. Decision Tree
    - Ensemble Method
        - Basic Ensemble Tech.
            - Max Voting
            - Averaging
            - Weighted Average
        - Advance Ensemble Tech.
            - Stacking
            - Blending
            - Bagging
            - Boosting
        - Algorithms based on Bagging an Boosting
            - Bagging Meta-estimator
            - Random Forest
            - AdaBoost
            - GBM
            - Light GBM
            - CatBoost
4. SVM
5. Naive Bayes
6. kNN
7. K-Means
8. Random Forest(bagging)
9. Dimensionality Reduction Algorithms
10. Gradient Boosting algorithms
    1. GBM
    2. XGBoost
    3. LightGBM
    4. CatBoost

#### 1. Linear Regression
It is used to estimate real values (cost of houses, number of calls, total sales etc.) based on continuous variable(s). Here, we establish relationship between independent and dependent variables by fitting a best line. This best fit line is known as regression line and represented by a linear equation 

$$Y= a * X + b$$

The best way to understand linear regression is to relive this experience of childhood. Let us say, you ask a child in fifth grade to arrange people in his class by increasing order of weight, without asking them their weights! What do you think the child will do? He / she would likely look (visually analyze) at the height and build of people and arrange them using a combination of these visible parameters. This is linear regression in real life! The child has actually figured out that height and build would be correlated to the weight by a relationship, which looks like the equation above.

In this equation:

- Y – Dependent Variable
- a – Slope
- X – Independent variable
- b – Intercept

These coefficients **a** and **b** are derived based on minimizing the sum of squared difference of distance between data points and regression line.

![Alt text](https://github.com/5267/ML/blob/master/resources/pics/Linear_Regression.png?raw=true)

Linear Regression is of mainly two types: Simple Linear Regression and Multiple Linear Regression. Simple Linear Regression is characterized by one independent variable. And, Multiple Linear Regression(as the name suggests) is characterized by multiple (more than 1) independent variables. While finding best fit line, you can fit a polynomial or curvilinear regression. And these are known as polynomial or curvilinear regression.

```
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# Load the diabetes dataset
diabetes = datasets.load_diabetes()


# Use only one feature
diabetes_X = diabetes.data[:, np.newaxis, 2]

# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# Split the targets into training/testing sets
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_y_train)

# Make predictions using the testing set
diabetes_y_pred = regr.predict(diabetes_X_test)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(diabetes_y_test, diabetes_y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(diabetes_y_test, diabetes_y_pred))

# Plot outputs
plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')
plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()
```

#### 2. Logistic Regression
It is used to estimate discrete values ( Binary values like 0/1, yes/no, true/false ) based on given set of independent variable(s). In simple words, it predicts the probability of occurrence of an event by fitting data to a logit function. Hence, it is also known as logit regression. Since, it predicts the probability, its output values lies between 0 and 1 (as expected).

$$\sigma(t) = \frac{1}{1+e^{-t}}=>p(x)=\frac{1}{1+e^{-(\beta_0+\beta_1x)}}$$

the case where **t** is a linear combination of multiple explanatory variables is treated similarly. the resulting expression for the probability **p(x)** ranges between 0 and 1.
![Alt text](https://github.com/5267/ML/blob/master/resources/pics/Logistic-curve.png?raw=true)

Suppose we wish to answer the following question:

    A group of 20 students spend between 0 and 6 hours studying for an exam. How does the number of hours spent studying affect the probability that the student will pass the exam?

The table shows the number of hours each student spent studying, and whether they passed (1) or failed (0).
<table class="wikitable">

<tbody><tr>
<th>Hours
</th>
<td>0.50</td>
<td>0.75</td>
<td>1.00</td>
<td>1.25</td>
<td>1.50</td>
<td>1.75</td>
<td>1.75</td>
<td>2.00</td>
<td>2.25</td>
<td>2.50</td>
<td>2.75</td>
<td>3.00</td>
<td>3.25</td>
<td>3.50</td>
<td>4.00</td>
<td>4.25</td>
<td>4.50</td>
<td>4.75</td>
<td>5.00</td>
<td>5.50
</td></tr>
<tr>
<th>Pass
</th>
<td>0</td>
<td>0</td>
<td>0</td>
<td>0</td>
<td>0</td>
<td>0</td>
<td>1</td>
<td>0</td>
<td>1</td>
<td>0</td>
<td>1</td>
<td>0</td>
<td>1</td>
<td>0</td>
<td>1</td>
<td>1</td>
<td>1</td>
<td>1</td>
<td>1</td>
<td>1
</td></tr></tbody></table>

The logistic regression analysis gives the following output
<table class="wikitable">
<tbody><tr>
<th></th>
<th>Coefficient</th>
<th>Std.Error</th>
<th>z-value</th>
<th>P-value (Wald)
</th></tr>
<tr style="text-align:right;">
<th>Intercept
</th>
<td>−4.0777</td>
<td>1.7610</td>
<td>−2.316</td>
<td>0.0206
</td></tr>
<tr style="text-align:right;">
<th>Hours
</th>
<td>1.5046</td>
<td>0.6287</td>
<td>2.393</td>
<td>0.0167
</td></tr></tbody></table>
The output indicates that hours studying is significantly associated with the probability of passing the exam (p=0.0167, p=0.0167}, Wald test)
For example, for a student who studies 4 hours, entering the value {Hours=2} in the equation gives the estimated probability of passing the exam of 0.87:

$$Probability of Passing Exam=\frac{1}{1+e^{(-(1.5046*4-4.0777))}}=0.87$$

```
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import datasets

# import some data to play with
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features.
Y = iris.target

logreg = LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial')

# Create an instance of Logistic Regression Classifier and fit the data.
logreg.fit(X, Y)

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
h = .02  # step size in the mesh
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1, figsize=(4, 3))
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', cmap=plt.cm.Paired)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())

plt.show()
```

#### 3、Decision Tree
Decision tree is a type of supervised learning algorithm (having a pre-defined target variable) that is mostly used in classification problems. It works for both categorical and continuous input and output variables. In this technique, we split the population or sample into two or more homogeneous sets (or sub-populations) based on most significant splitter / differentiator in input variables.

Let’s say we have a sample of 30 students with three variables Gender (Boy/ Girl), Class( IX/ X) and Height (5 to 6 ft). 15 out of these 30 play cricket in leisure time. Now, I want to create a model to predict who will play cricket during leisure period? In this problem, we need to segregate students who play cricket in their leisure time based on highly significant input variable among all three.

This is where decision tree helps, it will segregate the students based on all values of three variable and identify the variable, which creates the best homogeneous sets of students (which are heterogeneous to each other). In the snapshot below, you can see that variable Gender is able to identify best homogeneous sets compared to the other two variables.

![Alt text](https://github.com/5267/ML/blob/master/resources/pics/Decision%20Tree.png?raw=true)

#### 3.1 Types of Decision Trees

Types of decision tree is based on the type of target variable we have. It can be of two types:

1. Categorical Variable Decision Tree: Decision Tree which has categorical target variable then it called as categorical variable decision tree. **Example**:- In above scenario of student problem, where the target variable was “Student will play cricket or not” i.e. YES or NO.
2. Continuous Variable Decision Tree: Decision Tree has continuous target variable then it is called as Continuous Variable Decision Tree. **Example**:- Let’s say we have a problem to predict whether a customer will pay his renewal premium with an insurance company (yes/ no). Here we know that income of customer is a significant variable but insurance company does not have income details for all customers. Now, as we know this is an important variable, then we can build a decision tree to predict customer income based on occupation, product and various other variables. In this case, we are predicting values for continuous variable.

#### 3.2 How does a tree decide where to split?
The decision of making strategic splits heavily affects a tree’s accuracy. The decision criteria is different for classification and regression trees.

Decision trees use multiple algorithms to decide to split a node in two or more sub-nodes. The creation of sub-nodes increases the homogeneity of resultant sub-nodes. In other words, we can say that purity of the node increases with respect to the target variable. Decision tree splits the nodes on all available variables and then selects the split which results in most homogeneous sub-nodes.

The algorithm selection is also based on type of target variables. Let’s look at the four most commonly used algorithms in decision tree:

**3.2.1 Gini Index**

Gini index says, if we select two items from a population at random then they must be of same class and probability for this is 1 if population is pure.

1. It works with categorical target variable “Success” or “Failure”.
2. It performs only Binary splits
3. Higher the value of Gini higher the homogeneity.
4. CART (Classification and Regression Tree) uses Gini method to create binary splits.

**Steps to Calculate Gini for a split**
1. Calculate Gini for sub-nodes, using formula sum of square of probability for success and failure $p^2+q^2$.
2. Calculate Gini for split using weighted Gini score of each node of that split

**Example**: – Referring to example used above, where we want to segregate the students based on target variable ( playing cricket or not ). In the snapshot below, we split the population using two input variables Gender and Class. Now, I want to identify which split is producing more homogeneous sub-nodes using Gini index.

![Alt text](https://github.com/5267/ML/blob/master/resources/pics/Decision_Tree_Algorithm1.png?raw=true)

**Split on Gender**:
1. Calculate, Gini for sub-node Female = (0.2)*(0.2)+(0.8)*(0.8)=0.68
2. Gini for sub-node Male = (0.65)*(0.65)+(0.35)*(0.35)=0.55
3. Calculate weighted Gini for Split Gender = (10/30)*0.68+(20/30)*0.55 = **0.59**

**Similar for Split on Class**:
1. Gini for sub-node Class IX = (0.43)*(0.43)+(0.57)*(0.57)=0.51
2. Gini for sub-node Class X = (0.56)*(0.56)+(0.44)*(0.44)=0.51
3. Calculate weighted Gini for Split Class = (14/30)*0.51+(16/30)*0.51 = 0.51

Above, you can see that Gini score for Split on Gender is higher than Split on Class, hence, the node split will take place on Gender.

**3.2.2 Chi-Square**

It is an algorithm to find out the statistical significance between the differences between sub-nodes and parent node. We measure it by sum of squares of standardized differences between observed and expected frequencies of target variable.
1. It works with categorical target variable “Success” or “Failure”.
2. It can perform two or more splits.
3. Higher the value of Chi-Square higher the statistical significance of differences between sub-node and Parent node.
4. Chi-Square of each node is calculated using formula,
5. $Chi-square = \sqrt{\frac{(Actual – Expected)^2}{Expected}}$
6. It generates tree called CHAID (Chi-square Automatic Interaction Detector)

**Steps to Calculate Chi-square for a split:**
1. Calculate Chi-square for individual node by calculating the deviation for Success and Failure both
2. Calculated Chi-square of Split using Sum of all Chi-square of success and Failure of each node of the split

**Split on Gender:**
1. First we are populating for node Female, Populate the actual value for **“Play Cricket”** and **“Not Play Cricket”**, here these are 2 and 8 respectively.
2. Calculate expected value for **“Play Cricket”** and **“Not Play Cricket”**, here it would be 5 for both because parent node has probability of 50% and we have applied same probability on Female count(10).
3. Calculate deviations by using formula, Actual – Expected. It is for **“Play Cricket”** (2 – 5 = -3) and for **“Not play cricket”** ( 8 – 5 = 3).
4. Calculate Chi-square of node for **“Play Cricket”** and **“Not Play Cricket”** using formula with Chi-square formula above, You can refer below table for calculation.
5. Follow similar steps for calculating Chi-square value for Male node.
6. Now add all Chi-square values to calculate Chi-square for split Gender.

![Alt text](https://github.com/5267/ML/blob/master/resources/pics/Decision_Tree_Chi_Square1.png?raw=true)
Split on Class:

Perform similar steps of calculation for split on Class and you will come up with below table.
![Alt text](https://github.com/5267/ML/blob/master/resources/pics/Decision_Tree_Chi_Square_2.png?raw=true)

Above, you can see that Chi-square also identify the Gender split is more significant compare to Class.

**3.2.3 Information Gain**

Look at the image below and think which node can be described easily. I am sure, your answer is C because it requires less information as all values are similar. On the other hand, B requires more information to describe it and A requires the maximum information. In other words, we can say that C is a Pure node, B is less Impure and A is more impure.

![Alt text](https://github.com/5267/ML/blob/master/resources/pics/Information_Gain_Decision_Tree2.png?raw=true)

Now, we can build a conclusion that less impure node requires less information to describe it. And, more impure node requires more information. **Information theory** is a measure to define this degree of disorganization in a system known as **Entropy**. If the sample is completely homogeneous, then the entropy is zero and if the sample is an equally divided (50% – 50%), it has entropy of one.

Entropy can be calculated using formula:
$$Entropy = -p log_2p - q log_2q $$
Here *p* and *q* is probability of success and failure respectively in that node. Entropy is also used with categorical target variable. It chooses the split which has lowest entropy compared to parent node and other splits. The lesser the entropy, the better it is

**Steps to calculate entropy for a split**
1. Calculate entropy of parent node
2. Calculate entropy of each individual node of split and calculate weighted average of all sub-nodes available in split.

**Example**: Let’s use this method to identify best split for student example

1. Entropy for parent node = -(15/30) log2 (15/30) – (15/30) log2 (15/30) = 1. Here 1 shows that it is a impure node.
2. Entropy for Female node = -(2/10) log2 (2/10) – (8/10) log2 (8/10) = 0.72 and for male node,  -(13/20) log2 (13/20) – (7/20) log2 (7/20) = 0.93
3. Entropy for split Gender = Weighted entropy of sub-nodes = (10/30)*0.72 + (20/30)*0.93 = 0.86
4. Entropy for Class IX node, -(6/14) log2 (6/14) – (8/14) log2 (8/14) = 0.99 and for Class X node,  -(9/16) log2 (9/16) – (7/16) log2 (7/16) = 0.99.
5. Entropy for split Class =  (14/30)*0.99 + (16/30)*0.99 = 0.99

Above, you can see that entropy for Split on Gender is the lowest among all, so the tree will split on Gender. We can derive information gain from entropy as **1- Entropy**.

**3.2.4 Reduction in Variance**

Till now, we have discussed the algorithms for categorical target variable. Reduction in variance is an algorithm used for continuous target variables (regression problems). This algorithm uses the standard formula of variance to choose the best split. The split with lower variance is selected as the criteria to split the population:
$$Variance = \frac{\sum(X-\overline{X})^2}{n}$$
Above X-bar is mean of the values, X is actual and n is number of values.

**Steps to calculate Variance**:

Calculate variance for each node.
Calculate variance for each split as weighted average of each node variance.

**Example**:- Let’s assign numerical value 1 for play cricket and 0 for not playing cricket. Now follow the steps to identify the right split:

1. Variance for Root node, here mean value is (15*1 + 15*0)/30 = 0.5 and we have 15 one and 15 zero. Now variance would be ((1-0.5)^2+(1-0.5)^2+….15 times+(0-0.5)^2+(0-0.5)^2+…15 times) / 30, this can be written as (15*(1-0.5)^2+15*(0-0.5)^2) / 30 = 0.25
2. Mean of Female node =  (2*1+8*0)/10=0.2 and Variance = (2*(1-0.2)^2+8*(0-0.2)^2) / 10 = 0.16
3. Mean of Male Node = (13*1+7*0)/20=0.65 and Variance = (13*(1-0.65)^2+7*(0-0.65)^2) / 20 = 0.23
4. Variance for Split Gender = Weighted Variance of Sub-nodes = (10/30)*0.16 + (20/30) *0.23 = 0.21
5. Mean of Class IX node =  (6*1+8*0)/14=0.43 and Variance = (6*(1-0.43)^2+8*(0-0.43)^2) / 14= 0.24
6. Mean of Class X node =  (9*1+7*0)/16=0.56 and Variance = (9*(1-0.56)^2+7*(0-0.56)^2) / 16 = 0.25
7. Variance for Split Gender = (14/30)*0.24 + (16/30) *0.25 = 0.25

Above, you can see that Gender split has lower variance compare to parent node, so the split would take place on Gender variable.

#### 3.3 Overfitting challenges
Overfitting is one of the key challenges faced while modeling decision trees. If there is no limit set of a decision tree, it will give you 100% accuracy on training set because in the worse case it will end up making 1 leaf for each observation. Thus, preventing overfitting is pivotal while modeling a decision tree and it can be done in 2 ways:

1. Setting constraints on tree size
2. Tree pruning

**3.3.1 Setting constraints on tree size**

This can be done by using various parameters which are used to define a tree. First, lets look at the general structure of a decision tree:

![Alt text](https://github.com/5267/ML/blob/master/resources/pics/tree-infographic.png?raw=true)

The parameters used for defining a tree are further explained below

1. Minimum samples for a node split
    - Defines the minimum number of samples (or observations) which are required in a node to be considered for splitting.
    - Used to control over-fitting. Higher values prevent a model from learning relations which might be highly specific to the particular sample selected for a tree.
    - Too high values can lead to under-fitting hence, it should be tuned using **CV**.
2. Minimum samples for a terminal node (leaf)
    - Defines the minimum samples (or observations) required in a terminal node or leaf.
    - Used to control over-fitting similar to min_samples_split.
    - Generally lower values should be chosen for imbalanced class problems because the regions in which the minority class will be in majority will be very small.
3. Maximum depth of tree (vertical depth)
    - The maximum depth of a tree.
    - Used to control over-fitting as higher depth will allow model to learn relations very specific to a particular sample.
    - Should be tuned using **CV**.
4. Maximum number of terminal nodes
    - The maximum number of terminal nodes or leaves in a tree.
    - Can be defined in place of max_depth. Since binary trees are created, a depth of ‘n’ would produce a maximum of 2^n leaves.
5. Maximum features to consider for split
    - The number of features to consider while searching for a best split. These will be randomly selected.
    - As a thumb-rule, square root of the total number of features works great but we should check upto 30-40% of the total number of features.
    - Higher values can lead to over-fitting but depends on case to case.
 
 **3.2.2 Tree Pruning**

As discussed earlier, the technique of setting constraint is a greedy-approach. In other words, it will check for the best split instantaneously and move forward until one of the specified stopping condition is reached. Let’s consider the following case when you’re driving:


![Alt text](https://github.com/5267/ML/blob/master/resources/pics/tree-pruning.png?raw=true)

There are 2 lanes:

1. A lane with cars moving at 80km/h
2. A lane with trucks moving at 30km/h

At this instant, you are the yellow car and you have 2 choices:

1. Take a left and overtake the other 2 cars quickly
2. Keep moving in the present lane

Lets analyze these choice. In the former choice, you’ll immediately overtake the car ahead and reach behind the truck and start moving at 30 km/h, looking for an opportunity to move back right. All cars originally behind you move ahead in the meanwhile. This would be the optimum choice if your objective is to maximize the distance covered in next say 10 seconds. In the later choice, you sale through at same speed, cross trucks and then overtake maybe depending on situation ahead. Greedy you!

This is exactly the difference between **normal decision tree & pruning**. A decision tree with constraints won’t see the truck ahead and adopt a greedy approach by taking a left. On the other hand if we use pruning, we in effect look at a few steps ahead and make a choice.

So we know pruning is better. But how to implement it in decision tree? The idea is simple.

1. We first make the decision tree to a large depth.
2. Then we start at the bottom and start removing leaves which are giving us negative returns when compared from the top.
3. Suppose a split is giving us a gain of say -10 (loss of 10) and then the next split on that gives us a gain of 20. A simple decision tree will stop at step 1 but in pruning, we will see that the overall gain is +10 and keep both leaves.

#### 4、Working with Decision Trees in Python

```
# Import the necessary modules and libraries
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

# Create a random dataset
rng = np.random.RandomState(1)
X = np.sort(5 * rng.rand(80, 1), axis=0)
y = np.sin(X).ravel()
y[::5] += 3 * (0.5 - rng.rand(16))

# Fit regression model
regr_1 = DecisionTreeRegressor(max_depth=2)
regr_2 = DecisionTreeRegressor(max_depth=5)
regr_1.fit(X, y)
regr_2.fit(X, y)

# Predict
X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
y_1 = regr_1.predict(X_test)
y_2 = regr_2.predict(X_test)

# Plot the results
plt.figure()
plt.scatter(X, y, s=20, edgecolor="black",
            c="darkorange", label="data")
plt.plot(X_test, y_1, color="cornflowerblue",
         label="max_depth=2", linewidth=2)
plt.plot(X_test, y_2, color="yellowgreen", label="max_depth=5", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Decision Tree Regression")
plt.legend()
plt.show()
```

#### 5、What are ensemble methods in tree based modeling ?

The literary meaning of word ‘ensemble’ is **group**. Ensemble methods involve group of predictive models to achieve a better accuracy and model stability. Ensemble methods are known to impart supreme boost to tree based models.

Like every other model, a tree based model also suffers from the plague of bias and variance. **Bias** means, ‘how much on an average are the predicted values different from the actual value.’ **Variance** means, ‘how different will the predictions of the model be at the same point if different samples are taken from the same population’.
$$Error = Bias^2 + Variance + Noise$$
<font color='red'>注：Bias，刻画算法的拟合能力，偏高,表示预测函数与真实结果差异很大；Variance代表“同样大小的不同训练数据集训练出的模型”与这些模型的预期输出值“之间的差异，偏高，表示模型很不稳定；Noise是模型无法避免的噪音。</font>
You build a small tree and you will get a model with low variance and high bias. How do you manage to balance the trade off between bias and variance ?

Normally, as you increase the complexity of your model, you will see a reduction in prediction error due to lower bias in the model. As you continue to make your model more complex, you end up over-fitting your model and your model will start suffering from high variance.

A champion model should maintain a balance between these two types of errors. This is known as the **trade-off management** of bias-variance errors. Ensemble learning is one way to execute this trade off analysis.

![Alt text](https://github.com/5267/ML/blob/master/resources/pics/model_complexity.png?raw=true)

#### 5.1 Basic Ensemble Tech.
**5.1.1 Max Voting**
The max voting method is generally used for classification problems. In this technique, multiple models are used to make predictions for each data point. The predictions by each model are considered as a ‘vote’. The predictions which we get from the majority of the models are used as the final prediction.

For example, when you asked 5 of your colleagues to rate your movie (out of 5); we’ll assume three of them rated it as 4 while two of them gave it a 5. Since the majority gave a rating of 4, the final rating will be taken as 4. You can consider this as taking the mode of all the predictions.
![Alt text](https://github.com/5267/ML/blob/master/resources/pics/MaxVoting.png?raw=true)
**Sample Code**
```
model1 = tree.DecisionTreeClassifier()
model2 = KNeighborsClassifier()
model3= LogisticRegression()

model1.fit(x_train,y_train)
model2.fit(x_train,y_train)
model3.fit(x_train,y_train)

pred1=model1.predict(x_test)
pred2=model2.predict(x_test)
pred3=model3.predict(x_test)

final_pred = np.array([])
for i in range(0,len(x_test)):
    final_pred = np.append(final_pred, mode([pred1[i], pred2[i], pred3[i]]))
```

Alternatively, you can use “VotingClassifier” module in sklearn as follows:

```
from sklearn.ensemble import VotingClassifier
model1 = LogisticRegression(random_state=1)
model2 = tree.DecisionTreeClassifier(random_state=1)
model = VotingClassifier(estimators=[('lr', model1), ('dt', model2)], voting='hard')
model.fit(x_train,y_train)
model.score(x_test,y_test)
```

**5.1.2 Averaging**
Similar to the max voting technique, multiple predictions are made for each data point in averaging. In this method, we take an average of predictions from all the models and use it to make the final prediction. Averaging can be used for making predictions in regression problems or while calculating probabilities for classification problems.
![Alt text](https://github.com/5267/ML/blob/master/resources/pics/Averaging.png?raw=true)

**Sample Code**

```
model1 = tree.DecisionTreeClassifier()
model2 = KNeighborsClassifier()
model3= LogisticRegression()

model1.fit(x_train,y_train)
model2.fit(x_train,y_train)
model3.fit(x_train,y_train)

pred1=model1.predict_proba(x_test)
pred2=model2.predict_proba(x_test)
pred3=model3.predict_proba(x_test)

finalpred=(pred1+pred2+pred3)/3
```

**5.1.3 Weighted Average**
This is an extension of the averaging method. All models are assigned different weights defining the importance of each model for prediction. For instance, if two of your colleagues are critics, while others have no prior experience in this field, then the answers by these two friends are given more importance as compared to the other people.
![Alt text](https://github.com/5267/ML/blob/master/resources/pics/weighted%20Average.png?raw=true)

**Sample Code**

```
model1 = tree.DecisionTreeClassifier()
model2 = KNeighborsClassifier()
model3= LogisticRegression()

model1.fit(x_train,y_train)
model2.fit(x_train,y_train)
model3.fit(x_train,y_train)

pred1=model1.predict_proba(x_test)
pred2=model2.predict_proba(x_test)
pred3=model3.predict_proba(x_test)

finalpred=(pred1*0.3+pred2*0.3+pred3*0.4)
```

#### 5.2 Advance Ensemble Tech.
**5.2.1 Stacking**
Stacking is an ensemble learning technique that uses predictions from multiple models (for example decision tree, knn or svm) to build a new model. This model is used for making predictions on the test set. Below is a step-wise explanation for a simple stacked ensemble:

1. The train set is split into 10 parts.
![Alt text](https://github.com/5267/ML/blob/master/resources/pics/stacking1.png?raw=true)
2. A base model (suppose a decision tree) is fitted on 9 parts and predictions are made for the 10th part. This is done for each part of the train set.（注：分成几份，就会训练几次）
![Alt text](https://github.com/5267/ML/blob/master/resources/pics/stacking2.png?raw=true)
3. The base model (in this case, decision tree) is then fitted on the whole train dataset.
4. Using this model, predictions are made on the test set.
![Alt text](https://github.com/5267/ML/blob/master/resources/pics/stacking3.png?raw=true)
5. Steps 2 to 4 are repeated for another base model (say knn) resulting in another set of predictions for the train set and test set.
![Alt text](https://github.com/5267/ML/blob/master/resources/pics/stacking4.png?raw=true)
6. The predictions from the train set are used as features to build a new model.
![Alt text](https://github.com/5267/ML/blob/master/resources/pics/stacking5.png?raw=true)

This model is used to make final predictions on the test prediction set.

**Sample Code**
We first define a function to make predictions on n-folds of train and test dataset. This function returns the predictions for train and test for each model.

```
def Stacking(model,train,y,test,n_fold):
   folds=StratifiedKFold(n_splits=n_fold,random_state=1)
   test_pred=np.empty((test.shape[0],1),float)
   train_pred=np.empty((0,1),float)
   for train_indices,val_indices in folds.split(train,y.values):
      x_train,x_val=train.iloc[train_indices],train.iloc[val_indices]
      y_train,y_val=y.iloc[train_indices],y.iloc[val_indices]

      model.fit(X=x_train,y=y_train)
      train_pred=np.append(train_pred,model.predict(x_val))
      test_pred=np.append(test_pred,model.predict(test))
    return test_pred.reshape(-1,1),train_pred
```

Now we’ll create two base models – decision tree and knn.

```
model1 = tree.DecisionTreeClassifier(random_state=1)

test_pred1 ,train_pred1=Stacking(model=model1,n_fold=10, train=x_train,test=x_test,y=y_train)

train_pred1=pd.DataFrame(train_pred1)
test_pred1=pd.DataFrame(test_pred1)
```

```
model2 = KNeighborsClassifier()

test_pred2 ,train_pred2=Stacking(model=model2,n_fold=10,train=x_train,test=x_test,y=y_train)

train_pred2=pd.DataFrame(train_pred2)
test_pred2=pd.DataFrame(test_pred2)
```

Create a third model, logistic regression, on the predictions of the decision tree and knn models.

```
df = pd.concat([train_pred1, train_pred2], axis=1)
df_test = pd.concat([test_pred1, test_pred2], axis=1)

model = LogisticRegression(random_state=1)
model.fit(df,y_train)
model.score(df_test, y_test)
```

<font color='red'>注意事项：StratifiedKFold & KFold的区别</font>

**KFold交叉采样**：将训练/测试数据集划分n_splits个互斥子集，每次只用其中一个子集当做测试集，剩下的（n_splits-1）作为训练集，进行n_splits次实验并得到n_splits个结果。

**StratifiedKFold分层采样**：用于交叉验证：与KFold最大的差异在于，StratifiedKFold方法是根据标签中不同类别占比来进行拆分数据的，分层采样，确保训练集，测试集中各类别样本的**比例与原始数据集中相同**。

实例分析两者差别
```
1、首先生成8行数据(含特征和标签数据)
import numpy as np
from sklearn.model_selection import StratifiedKFold,KFold

X=np.array([
    [1,2,3,4],
    [11,12,13,14],
    [21,22,23,24],
    [31,32,33,34],
    [41,42,43,44],
    [51,52,53,54],
    [61,62,63,64],
    [71,72,73,74]
])
 
y=np.array([1,1,0,0,1,1,0,0])
```
```
2、利用KFold方法交叉采样：按顺序分别取第1-2、3-4、5-6和7-8的数据
kfolder = KFold(n_splits=4,random_state=1)
for train, test in kfolder.split(X,y):
    print('Train: %s | test: %s' % (train, test),'\n')
>>>注意下面的234是数据的标签
Train: [2 3 4 5 6 7] | test: [0 1]
Train: [0 1 4 5 6 7] | test: [2 3]
Train: [0 1 2 3 6 7] | test: [4 5]
Train: [0 1 2 3 4 5] | test: [6 7]
```

```
3、利用StratifiedKFold方法分层采样：依照标签的比例来抽取数据，本案例集标签0和1的比例是1：1，因此在抽取数据时也是按照标签比例1：1来提取的
sfolder = StratifiedKFold(n_splits=4,random_state=0)
for train, test in sfolder.split(X,y):
    print('Train: %s | test: %s' % (train, test))
>>>
Train: [1 3 4 5 6 7] | test: [0 2]
Train: [0 2 4 5 6 7] | test: [1 3]
Train: [0 1 2 3 5 7] | test: [4 6]
Train: [0 1 2 3 4 6] | test: [5 7]

```

**5.2.2 Blending**
Blending follows the same approach as stacking but uses only a holdout (validation) set from the train set to make predictions. In other words, unlike stacking, the predictions are made on the holdout set only.

1. The train set is split into training and validation sets.
![Alt text](https://github.com/5267/ML/blob/master/resources/pics/blending1.png?raw=true)

2. Model(s) are fitted on the training set.
3. The predictions are made on the validation set and the test set.
![Alt text](https://github.com/5267/ML/blob/master/resources/pics/blending2.png?raw=true)

4. The validation set and its predictions are used as features to build a new model.
5. This model is used to make final predictions on the test and meta-features.

**Sample Code**

We’ll build two models, decision tree and knn, on the train set in order to make predictions on the validation set.

```
model1 = tree.DecisionTreeClassifier()
model1.fit(x_train, y_train)
val_pred1=model1.predict(x_val)
test_pred1=model1.predict(x_test)
val_pred1=pd.DataFrame(val_pred1)
test_pred1=pd.DataFrame(test_pred1)

model2 = KNeighborsClassifier()
model2.fit(x_train,y_train)
val_pred2=model2.predict(x_val)
test_pred2=model2.predict(x_test)
val_pred2=pd.DataFrame(val_pred2)
test_pred2=pd.DataFrame(test_pred2)
```

Combining the meta-features and the validation set, a logistic regression model is built to make predictions on the test set.

```
df_val=pd.concat([x_val, val_pred1,val_pred2],axis=1)
df_test=pd.concat([x_test, test_pred1,test_pred2],axis=1)

model = LogisticRegression()
model.fit(df_val,y_val)
model.score(df_test,y_test)
```

**5.2.3 Bagging**
**5.2.4 Boosting**
#### 5.3 Algorithms based on Bagging an Boosting

- Bagging Meta-estimator

- Random Forest
- AdaBoost
- GBM
- Light GBM
- CatBoost

Some of the commonly used ensemble methods include: **Bagging, Boosting and Stacking**

#### 5.1 bagging, How does it work?
Bagging is a technique used to reduce the variance of our predictions by combining the result of multiple classifiers modeled on different sub-samples of the same data set. The following figure will make it clearer:
![Alt text](https://github.com/5267/ML/blob/master/resources/pics/bagging.png?raw=true)

The steps followed in bagging are:

1. Create Multiple DataSets:
    - Sampling is done with replacement on the original data and new datasets are formed.
    - The new data sets can have a fraction of the columns as well as rows, which are generally hyper-parameters in a bagging model
    - Taking row and column fractions less than 1 helps in making robust models, less prone to overfitting
2. Build Multiple Classifiers:
    - Classifiers are built on each data set.
    - Generally the same classifier is modeled on each data set and predictions are made.
3. Combine Classifiers:
    - The predictions of all the classifiers are combined using a **mean, median or mode** value depending on the problem at hand.
    - The combined values are generally more robust than a single model.

Note that, here the number of models built is not a hyper-parameters. Higher number of models are always better or may give similar performance than lower numbers. It can be theoretically shown that the variance of the combined predictions are reduced to 1/n (n: number of classifiers) of the original variance, under some assumptions.

There are various implementations of bagging models. Random forest is one of them.

#### 5.1.1 Random Forest
Random Forest is considered to be a panacea of all data science problems. On a funny note, when you can’t think of any algorithm (irrespective of situation), use random forest!

Random Forest is a versatile machine learning method capable of performing both regression and classification tasks. It also undertakes dimensional reduction methods, treats missing values, outlier values and other essential **[steps of data exploration](https://www.analyticsvidhya.com/blog/2015/02/data-exploration-preparation-model/)**, and does a fairly good job. It is a type of ensemble learning method, where a group of weak models combine to form a powerful model.

**How does it work?**
In Random Forest, we grow multiple trees as opposed to a single tree in CART model (see comparison between CART and Random Forest here, [part1](https://www.analyticsvidhya.com/blog/2014/06/comparing-cart-random-forest-1/) and [part2](https://www.analyticsvidhya.com/blog/2014/06/comparing-random-forest-simple-cart-model/))

To classify a new object based on attributes, each tree gives a classification and we say the tree “votes” for that class. The forest chooses the classification having the most votes (over all the trees in the forest) and in case of regression, it takes the average of outputs by different trees.

It works in the following manner. Each tree is planted & grown as follows:

1. Assume number of cases in the training set is N. Then, sample of these N cases is taken at random but with replacement. This sample will be the training set for growing the tree.
2. If there are M input variables, a number m<M is specified such that at each node, m variables are selected at random out of the M. The best split on these m is used to split the node. The value of m is held constant while we grow the forest.
3. Each tree is grown to the largest extent possible and  there is no pruning.
4. Predict new data by aggregating the predictions of the ntree trees (i.e., majority votes for classification, average for regression).

**Case Study**

Following is a distribution of Annual income Gini Coefficients across different countries :
![Alt text](https://github.com/5267/ML/blob/master/resources/pics/oecd-income_inequality_2013_2.png?raw=true)

Mexico has the second highest Gini coefficient and hence has a very high segregation in annual income of rich and poor. Our task is to come up with an accurate predictive algorithm to estimate annual income bracket of each individual in Mexico. The brackets of income are as follows :

1. Below $40,000
2. $40,000 – 150,000
3. More than $150,000

Following are the information available for each individual :

1. Age , 
2. Gender,  
3. Highest educational qualification, 
4. Working in Industry, 
5. Residence in Metro/Non-metro

We need to come up with an algorithm to give an accurate prediction for an individual who has following traits:

1. Age : 35 years , 
2, Gender : Male , 
3. Highest Educational Qualification : Diploma holder, 
4. Industry : Manufacturing, 
5. Residence : Metro

**The algorithm  of Random Forest**

**Random forest** is like bootstrapping algorithm with Decision tree (CART) model. Say, we have **1000 observation** in the complete population with **10 variables**. Random forest tries to build multiple CART model with different sample and different initial variables. For instance, it will take a random sample of 100 observation and 5 randomly chosen initial variables to build a CART model. It will repeat the process (say) 10 times and then make a final prediction on each observation. Final prediction is a function of each prediction. This final prediction can simply be the mean of each prediction.

**Back to Case  study**

Mexico has a population of 118 MM. Say, the algorithm Random forest picks up 10k observation with only one variable (for simplicity) to build each CART model. In total, we are looking at 5 CART model being built with different variables. In a real life problem, you will have more number of population sample and different combinations of  input variables.

**Salary bands** :

1. Band 1 : Below $40,000
2. Band 2: $40,000 – 150,000
3. Band 3: More than $150,000

Following are the outputs of the 5 different CART model.

CART 1 : Variable Age
![Alt text](https://github.com/5267/ML/blob/master/resources/pics/rf1.png?raw=true)

CART 2 : Variable Gender
![Alt text](https://github.com/5267/ML/blob/master/resources/pics/rf2.png?raw=true)

CART 3 : Variable Education
![Alt text](https://github.com/5267/ML/blob/master/resources/pics/rf3.png?raw=true)

CART 4 : Variable Residence
![Alt text](https://github.com/5267/ML/blob/master/resources/pics/rf4.png?raw=true)

CART 5 : Variable Industry
![Alt text](https://github.com/5267/ML/blob/master/resources/pics/rf5.png?raw=true)

Using these 5 CART models, we need to come up with singe set of probability to belong to each of the salary classes. For simplicity, we will just take a mean of probabilities in this case study. Other than simple mean, we also consider vote method to come up with the final prediction. To come up with the final prediction let’s locate the following profile in each CART model :

1. Age : 35 years , 
2. Gender : Male , 
3. Highest Educational Qualification : Diploma holder, 
4. Industry : Manufacturing, 
5. Residence : Metro

For each of these CART model, following is the distribution across salary bands :
![Alt text](https://github.com/5267/ML/blob/master/resources/pics/DF.png?raw=true)

The final probability is simply the average of the probability in the same salary bands in different CART models. As you can see from this analysis, that there is 70% chance of this individual falling in class 1 (less than $40,000) and around 24% chance of the individual falling in class 2.

**Advantages of Random Forest**

- This algorithm can solve both type of problems i.e. classification and regression and does a decent estimation at both fronts.
- One of benefits of Random forest which excites me most is, the power of handle large data set with higher dimensionality. It can handle thousands of input variables and identify most significant variables so it is considered as one of the dimensionality reduction methods. Further, the model outputs **Importance of variable**, which can be a very handy feature (on some random data set).

![Alt text](
https://github.com/5267/ML/blob/master/resources/pics/Variable_Important.png?raw=true)

- It has an effective method for estimating missing data and maintains accuracy when a large proportion of the data are missing.
- It has methods for balancing errors in data sets where classes are imbalanced.
- The capabilities of the above can be extended to unlabeled data, leading to unsupervised clustering, data views and outlier detection.
- Random Forest involves sampling of the input data with replacement called as bootstrap sampling. Here one third of the data is not used for training and can be used to testing. These are called the out of bag samples. Error estimated on these **out of bag** samples is known as out of bag error. Study of error estimates by Out of bag, gives evidence to show that the out-of-bag estimate is as accurate as using a test set of the same size as the training set. Therefore, using the out-of-bag error estimate removes the need for a set aside test set.

**Disadvantages of Random Forest**

- It surely does a good job at classification but not as good as for regression problem as it does not give precise continuous nature predictions. In case of regression, it doesn’t predict beyond the range in the training data, and that they may over-fit data sets that are particularly noisy.
- Random Forest can feel like a black box approach for statistical modelers – you have very little control on what the model does. You can at best – try different parameters and random seeds!

```
#Import Library
from sklearn.ensemble import RandomForestClassifier #use RandomForestRegressor for regression problem
#Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
# Create Random Forest object
model= RandomForestClassifier(n_estimators=1000)
# Train the model using the training sets and check score
model.fit(X, y)
#Predict Output
predicted= model.predict(x_test)
```

#### 5.2 Boosting, How does it work?
<u>Definition:</u> The term ‘Boosting’ refers to a family of algorithms which converts **weak** learner to **strong** learners.

**Boost vs Bagging**
Boosting algorithms play a crucial role in dealing with bias variance trade-off. Unlike bagging algorithms, which **only controls for high variance** in a model, boosting controls both the aspects (bias & variance), and is considered to be more effective. 

Let’s understand this definition in detail by solving a problem of spam email identification:

How would you classify an email as SPAM or not? Like everyone else, our initial approach would be to identify ‘spam’ and ‘not spam’ emails using following criteria. If:

1. Email has only one image file (promotional image), It’s a SPAM
2. Email has only link(s), It’s a SPAM
3. Email body consist of sentence like “You won a prize money of $ xxxxxx”, It’s a SPAM
4. Email from our official domain “Analyticsvidhya.com” , Not a SPAM
5. Email from known source, Not a SPAM

Above, we’ve defined multiple rules to classify an email into ‘spam’ or ‘not spam’. But, do you think these rules individually are strong enough to successfully classify an email? No.

Individually, these rules are not powerful enough to classify an email into ‘spam’ or ‘not spam’. Therefore, these rules are called as weak learner.

To convert weak learner to strong learner, we’ll combine the prediction of each weak learner using methods like:

- Using average/ weighted average
- Considering prediction has higher vote

For example:  Above, we have defined 5 weak learners. Out of these 5, 3 are voted as ‘SPAM’ and 2 are voted as ‘Not a SPAM’. In this case, by default, we’ll consider an email as SPAM because we have higher(3) vote for ‘SPAM’.

**How does it work ?**

combines a set of weak learners and delivers improved prediction accuracy. At any instant t, the model outcomes are weighed based on the outcomes of previous instant t-1. The outcomes **predicted correctly** are given a **lower weight** and the ones miss-classified are weighted higher.

Let’s understand it visually:
![Alt text](https://github.com/5267/ML/blob/master/resources/pics/boosting.png?raw=true)

1. Box 1: Output of First Weak Learner (From the left)
Initially all points have same weight (denoted by their size).
The decision boundary predicts 2 +ve and 5 -ve points correctly.

2. Box 2: Output of Second Weak Learner
The points classified correctly in box 1 are given a lower weight and vice versa.
The model focuses on high weight points now and classifies them correctly. But, others are misclassified now.

Similar trend can be seen in box 3 as well. This continues for many iterations. In the end, all models are given a weight depending on their accuracy and a consolidated result is generated.

Finally, it combines the outputs from weak learner and creates  a strong learner which eventually improves the prediction power of the model. Boosting pays higher focus on examples which are mis-classiﬁed or have higher errors by preceding weak rules.
There are many boosting algorithms which impart additional boost to model’s accuracy. In this tutorial, we’ll learn about the two most commonly used algorithms i.e. **AdaBoost（Adaptive Boosting）、Gradient Boosting (GBM) and XGboost**.

#### 5.2.1 AdaBoost
<font color='red'>Classification Boosters</font>：Let’s take a very simple example to understand the underlying concept of AdaBoost. You have two classes : 0’s and 1’s. Each number is an observation. The only two features available is x-axis and y-axis. For instance (1,1)  is a 0 while (4,4) is a 1. Now using these two features you need to classify each observation. Our ultimate objective remains the same as any classifier problem : find the classification boundary. Following are the step we follow to apply an AdaBoost.

**Step 1 : Visualize the data** : Let’s first understand the data and find insights on whether we have a linear classifier boundary. As shown below, no such boundary exist which can separate 0’s from 1’s.
![Alt text](https://github.com/5267/ML/blob/master/resources/pics/adboost1.png?raw=true)

**Step 2 : Make the first Decision stump**: You have already read about decision trees in many of our previous articles. Decision stump is a unit depth tree which decides just 1 most significant cut on features. Here it chooses draw the boundary starting from the third row from top. Now the yellow portion is expected to be all 0’s and unshaded portion to be all 1’s. However, we see high number of false positive post we build this decision stump. We have nine 1’s being wrongly qualified as 0’s. And similarly eighteen 0’s qualified as 1’s.
![Alt text](https://github.com/5267/ML/blob/master/resources/pics/adboost2.png?raw=true)

**Step 3 : Give additional weight to mis-classified observations**: Once we know the misclassified observations, we give additional weight to these observations. Hence, you see 0’s and 1’s in bold which were misclassified before. In the next level, we will make sure that these highly weighted observation are classified correct
![Alt text](https://github.com/5267/ML/blob/master/resources/pics/adboost3.png?raw=true)

**Step 4 : Repeat the process and combine all stumps to get final classifier** : We repeat the process multiple times and focus more on previously misclassified observations. Finally, we take a weighted mean of all the boudaries discovered which will look something as below.
![Alt text](https://github.com/5267/ML/blob/master/resources/pics/adboost4.png?raw=true)

Real life examples：[Face Detection](https://www.analyticsvidhya.com/blog/2015/01/basics-image-processing-feature-extraction-python/)

```
from sklearn.ensemble import AdaBoostClassifier #For Classification
from sklearn.ensemble import AdaBoostRegressor #For Regression
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier() 
clf = AdaBoostClassifier(n_estimators=100, base_estimator=dt,learning_rate=1)
#Above I have used decision tree as a base estimator, you can use any ML learner as base estimator if it ac# cepts sample weight 
clf.fit(x_train,y_train)
```

#### 5.2.2 Gradient Boosting (GBM)
