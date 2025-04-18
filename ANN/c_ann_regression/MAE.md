### Why Use MAE (Mean Absolute Error) as a Metric?

When building a regression model, we typically use a metric to evaluate how well the model is performing by comparing its predictions against the true values. **MAE (Mean Absolute Error)** is a **common evaluation metric** for regression tasks because of the following reasons:

#### 1. **Intuitive Interpretation**: 
   - MAE represents the **average magnitude** of the errors in the model’s predictions, **without considering direction** (positive or negative). In simpler terms, it tells you how far off your predictions are, on average, from the actual values.
   - It is measured in the same units as the target variable (in this case, MPa), which makes it easy to interpret. For example, if MAE = 5, it means that, on average, the model's predictions are off by 5 MPa.

#### 2. **Robustness to Outliers**:
   - MAE **doesn’t amplify large errors** like Mean Squared Error (MSE) does. This makes MAE a good choice when we care about **robust performance across all samples**, and when we don't want large errors to disproportionately influence the model's performance.
   - In MSE, large errors get squared, which can lead to a model that focuses on reducing extreme outliers at the expense of more typical data points. With MAE, each error is treated equally, regardless of whether it's small or large.

#### 3. **Simple to Calculate**:
   - The formula for MAE is **straightforward** and easy to compute. It gives you a direct sense of how far the model's predictions are from the ground truth.
   
#### 4. **Good for Interpretable Loss**:
   - When you use MAE as a metric, you can easily relate the number to the real-world problem you're solving. For example, if you're predicting the compressive strength of concrete, an MAE of 2 MPa means that on average, the model’s predictions are 2 MPa away from the true value.

---

### How Does MAE Work?

MAE is defined as the **average absolute difference** between the predicted values and the actual values. The formula for MAE is:

\[
\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
\]

Where:
- \( n \) = the number of data points (samples).
- \( y_i \) = the actual (true) value of the \( i \)-th sample.
- \( \hat{y}_i \) = the predicted value for the \( i \)-th sample.
- \( |y_i - \hat{y}_i| \) = the **absolute error** for each sample.

#### **Steps to Calculate MAE**:
1. **For each sample**, calculate the absolute difference between the predicted value (\( \hat{y}_i \)) and the actual value (\( y_i \)).

   - \( |y_i - \hat{y}_i| \) ensures that we always get positive errors, regardless of whether the model’s prediction is higher or lower than the actual value.
   
2. **Sum all the absolute errors** for all data points.
   
3. **Divide by the total number of samples** to get the **average absolute error**.

---

### Example: MAE Calculation

Let’s consider a small example with 5 data points:

| Actual (True Value) \( y_i \) | Predicted Value \( \hat{y}_i \) | Absolute Error \( |y_i - \hat{y}_i| \) |
|-------------------------------|-------------------------------|-------------------|
| 10                            | 8                             | 2                 |
| 20                            | 22                            | 2                 |
| 30                            | 27                            | 3                 |
| 40                            | 43                            | 3                 |
| 50                            | 48                            | 2                 |

Now, we calculate MAE:

1. **Absolute errors** for each sample:
   - \( |10 - 8| = 2 \)
   - \( |20 - 22| = 2 \)
   - \( |30 - 27| = 3 \)
   - \( |40 - 43| = 3 \)
   - \( |50 - 48| = 2 \)

2. **Sum of absolute errors**:  
   \( 2 + 2 + 3 + 3 + 2 = 12 \)

3. **MAE**:
   \[
   \text{MAE} = \frac{12}{5} = 2.4
   \]

So, the MAE in this example is **2.4**, meaning on average, the model's predictions are off by **2.4 units** (in whatever units your target variable is, like MPa).

---

### **Advantages and Disadvantages of MAE**

#### ✅ Advantages:
- **Interpretability**: MAE gives you an easily interpretable result (average error in the same units as the target).
- **No large error penalty**: Unlike MSE, MAE treats all errors equally, which can be beneficial when you don’t want to disproportionately penalize large errors (e.g., if they might be outliers).
- **Less sensitive to outliers**: MAE won’t "blow up" for large outliers, making it more **robust** when your dataset contains extreme values.

#### ❌ Disadvantages:
- **Not differentiable at 0**: MAE's absolute error function isn't differentiable at zero, which can make it more challenging for optimization algorithms like gradient descent to converge smoothly. However, this is generally not a major issue because it’s still widely used in practice.
- **Less sensitive to large errors**: If you're concerned about large errors in your predictions (for example, predicting compressive strength as 100 MPa when the true value is 10 MPa), MAE won’t penalize these large errors as heavily as MSE would. 

---

### **In Summary:**
- **MAE** is a good metric for regression tasks when you want a **simple, interpretable measure of average error** without giving extra weight to large outliers. It's easy to calculate and is **directly interpretable in the same units as the target variable**.
- It is useful when you want to treat all errors equally, instead of emphasizing large errors as **MSE** would.


---

Lets explore **MSE (Mean Squared Error)**, **RMSE (Root Mean Squared Error)**, and **R-squared (Coefficient of Determination)**. This shall give a clearer idea of how to evaluate and select the best metric for regression model. Let’s walk through each of these metrics and how you can decide which one to use based on specific needs.

---

### 1. **MSE (Mean Squared Error)**

**Definition**: 
MSE is the average of the squared differences between predicted values (\( \hat{y} \)) and actual values (\( y \)).

\[
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
\]

#### **How it works**:
- **Squaring the errors** means that large errors (deviations) will have a **greater impact** on the MSE, because they are squared.
- This makes MSE **sensitive to outliers** and large errors.
- It’s measured in the **square of the units of the target variable**. For example, if you're predicting the compressive strength of concrete (MPa), the MSE will be in **MPa²**, which can be harder to interpret directly.

#### **Advantages**:
- **Sensitive to large errors**: MSE penalizes large deviations more than small ones, which is useful if you care about reducing large errors.
- **Mathematically tractable**: Since MSE is differentiable, it's often a good choice when training models using gradient-based optimization techniques (like gradient descent).

#### **Disadvantages**:
- **Harder to interpret**: Since MSE is in the squared units, it's not directly interpretable in the same units as the target variable. For example, an MSE of 16 doesn't tell you much about the actual magnitude of the error.
- **Sensitive to outliers**: A few large errors can heavily influence the overall metric.

---

### 2. **RMSE (Root Mean Squared Error)**

**Definition**: 
RMSE is the square root of MSE, which makes it interpretable in the **same units** as the target variable.

\[
\text{RMSE} = \sqrt{\text{MSE}} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
\]

#### **How it works**:
- RMSE is simply the **square root of MSE**.
- It makes the metric more interpretable because it is in the **same units** as the target variable.
- Like MSE, RMSE gives more weight to large errors because it is derived from squaring the errors.

#### **Advantages**:
- **Interpretable in the same units**: RMSE is easier to understand and interpret because it is in the same units as the target variable.
- **Sensitive to large errors**: Like MSE, RMSE penalizes large errors, making it useful when large errors are a concern.

#### **Disadvantages**:
- **Sensitive to outliers**: Just like MSE, RMSE is **sensitive to large errors**, so a few extreme errors can significantly increase the RMSE.
- **Less robust for certain applications**: If your goal is to model without overemphasizing large errors, RMSE might not be the best choice.

---

### 3. **R-squared (Coefficient of Determination)**

**Definition**: 
R-squared measures how well the model's predictions fit the data by comparing the total variation in the actual values to the variation explained by the model.

\[
R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}
\]

Where:
- \( \hat{y}_i \) is the predicted value for the \( i \)-th sample.
- \( y_i \) is the true value.
- \( \bar{y} \) is the mean of the true values.

#### **How it works**:
- **R-squared compares the variance explained by the model** to the total variance in the data.
- A **higher R-squared** value (close to 1) indicates that the model explains a **large proportion of the variance** in the target variable.
- A **low or negative R-squared** indicates that the model is performing poorly or that it is worse than simply predicting the mean value of the target.

#### **Advantages**:
- **Interpretability**: R-squared is easy to interpret. A value of 0.85 means that 85% of the variance in the target variable is explained by the model.
- **Scale-independent**: R-squared is not affected by the scale of the target variable.

#### **Disadvantages**:
- **Not sensitive to outliers**: R-squared doesn’t directly penalize large errors as MSE or RMSE do. It focuses more on the model’s ability to explain variance.
- **Can be misleading**: A high R-squared value doesn’t always indicate a good model. For example, you could get a high R-squared value even if your model is overfitting the data. R-squared alone doesn’t tell you if your model is generalizing well to new data.
- **Doesn’t work well for non-linear regression**: R-squared is more suited to linear regression models, and it may not reflect true performance in non-linear models.

---

### **How to Choose the Right Metric:**

#### **1. If you care about overall error magnitude:**
- **Use MAE** or **RMSE**.
- **MAE** is less sensitive to outliers and gives a straightforward understanding of average error in the target units.
- **RMSE** is good when you want to **penalize large errors more** and have a metric in the same units as the target variable. It is better when large errors matter more.

#### **2. If you want a metric that punishes large errors more:**
- **Use MSE** or **RMSE**.
- Both MSE and RMSE are sensitive to outliers, so they will increase if large errors are made. If large errors should be penalized heavily (e.g., in some business or financial applications), these metrics are a good choice.

#### **3. If you want to evaluate the proportion of variance explained:**
- **Use R-squared**.
- R-squared is great for understanding how much of the target variable's variance is explained by your model. However, it may not be enough on its own because it doesn't penalize large errors or tell you about the **actual magnitude** of errors.

#### **4. If you want a scale-independent metric:**
- **Use R-squared** or **Normalized RMSE**.
- **R-squared** is not affected by the scale of your target variable, which can be useful when comparing models across different datasets with different ranges.

#### **5. If you’re dealing with non-linear relationships:**
- **Avoid relying solely on R-squared** for non-linear models, and prefer **MAE**, **RMSE**, or even custom metrics that reflect the error distribution better.

---

### **Example of Use Case Selection:**

Let’s say you’re modeling concrete compressive strength:
- **If predicting large deviations (i.e., outliers) matters**, you would prefer **RMSE** because it punishes large errors more.
- **If you want a clear interpretation of how wrong the model’s predictions are on average** (in terms of actual units, like MPa), **MAE** would be ideal.
- **If you want to compare how much variance your model explains**, and you’re not worried about individual error magnitudes, **R-squared** is appropriate.

---

### Conclusion:

- **MAE** is simple, interpretable, and robust to outliers.
- **MSE** and **RMSE** are more sensitive to large errors and are useful if you care about reducing big mistakes in predictions.
- **R-squared** is best when you want to evaluate the goodness of fit of your model, but should be used with caution, especially in cases of non-linear models or models prone to overfitting.

Ultimately, **the choice of metric depends on the specific goals of your model**. You might even want to evaluate your model using a combination of these metrics to get a more holistic view of its performance.



