---
jupyter:
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.7.12
  nbformat: 4
  nbformat_minor: 4
---

::: {.cell .code execution_count="148" _cell_guid="b1076dfc-b9ad-4769-8c92-a6c4dae69d19" _uuid="8f2839f25d086af736a60e9eeb907d3b93b6e0e5" execution="{\"iopub.execute_input\":\"2023-01-15T03:12:07.857338Z\",\"iopub.status.busy\":\"2023-01-15T03:12:07.856820Z\",\"iopub.status.idle\":\"2023-01-15T03:12:07.868148Z\",\"shell.execute_reply\":\"2023-01-15T03:12:07.866766Z\",\"shell.execute_reply.started\":\"2023-01-15T03:12:07.857296Z\"}" trusted="true"}
``` python
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
```

::: {.output .stream .stdout}
    /kaggle/input/star-dataset/6 class csv.csv
:::
:::

::: {.cell .markdown}
# In this dataset I wanted to make a Machine Learning model to predict star type from the other variables, and learn more about data science concepts along the way

# Made possible thanks to the below sources:

<https://jayant017.medium.com/hyperparameter-tuning-in-xgboost-using-randomizedsearchcv-88fcb5b58a73>

<https://www.youtube.com/watch?v=ap2SS0-XPcE>

<https://www.kaggle.com/learn/intermediate-machine-learning>
:::

::: {.cell .markdown}
# Begin importing dataset and printing head
:::

::: {.cell .code execution_count="149" execution="{\"iopub.execute_input\":\"2023-01-15T03:12:07.871783Z\",\"iopub.status.busy\":\"2023-01-15T03:12:07.871296Z\",\"iopub.status.idle\":\"2023-01-15T03:12:07.894963Z\",\"shell.execute_reply\":\"2023-01-15T03:12:07.893429Z\",\"shell.execute_reply.started\":\"2023-01-15T03:12:07.871732Z\"}" trusted="true"}
``` python
df = pd.read_csv('/kaggle/input/star-dataset/6 class csv.csv')
df.head()
```

::: {.output .execute_result execution_count="149"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Temperature (K)</th>
      <th>Luminosity(L/Lo)</th>
      <th>Radius(R/Ro)</th>
      <th>Absolute magnitude(Mv)</th>
      <th>Star type</th>
      <th>Star color</th>
      <th>Spectral Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3068</td>
      <td>0.002400</td>
      <td>0.1700</td>
      <td>16.12</td>
      <td>0</td>
      <td>Red</td>
      <td>M</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3042</td>
      <td>0.000500</td>
      <td>0.1542</td>
      <td>16.60</td>
      <td>0</td>
      <td>Red</td>
      <td>M</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2600</td>
      <td>0.000300</td>
      <td>0.1020</td>
      <td>18.70</td>
      <td>0</td>
      <td>Red</td>
      <td>M</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2800</td>
      <td>0.000200</td>
      <td>0.1600</td>
      <td>16.65</td>
      <td>0</td>
      <td>Red</td>
      <td>M</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1939</td>
      <td>0.000138</td>
      <td>0.1030</td>
      <td>20.06</td>
      <td>0</td>
      <td>Red</td>
      <td>M</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {.cell .markdown}
# I was interested in seeing some bivariate correlations, used scatterplot to see any relationships
:::

::: {.cell .code execution_count="150" execution="{\"iopub.execute_input\":\"2023-01-15T03:12:07.898075Z\",\"iopub.status.busy\":\"2023-01-15T03:12:07.897060Z\",\"iopub.status.idle\":\"2023-01-15T03:12:08.139460Z\",\"shell.execute_reply\":\"2023-01-15T03:12:08.138051Z\",\"shell.execute_reply.started\":\"2023-01-15T03:12:07.898021Z\"}" trusted="true"}
``` python
df.plot.scatter(x = 'Temperature (K)', y = 'Spectral Class')
```

::: {.output .execute_result execution_count="150"}
    <AxesSubplot:xlabel='Temperature (K)', ylabel='Spectral Class'>
:::

::: {.output .display_data}
![](vertopal_2eb67dc2b1c046c6ae9007e8cc5ee010/742e52bf5599c8c235a85863324053b0980ffe03.png)
:::
:::

::: {.cell .code execution_count="151" execution="{\"iopub.execute_input\":\"2023-01-15T03:12:08.143252Z\",\"iopub.status.busy\":\"2023-01-15T03:12:08.142088Z\",\"iopub.status.idle\":\"2023-01-15T03:12:08.394786Z\",\"shell.execute_reply\":\"2023-01-15T03:12:08.393078Z\",\"shell.execute_reply.started\":\"2023-01-15T03:12:08.143196Z\"}" trusted="true"}
``` python
df.plot.scatter(x = 'Temperature (K)', y = 'Star type')
```

::: {.output .execute_result execution_count="151"}
    <AxesSubplot:xlabel='Temperature (K)', ylabel='Star type'>
:::

::: {.output .display_data}
![](vertopal_2eb67dc2b1c046c6ae9007e8cc5ee010/73f0c0c427a6fabf8818222f9b5ac2a95da64ceb.png)
:::
:::

::: {.cell .markdown}
# Radius vs Star Type seems to have an emergent pattern, an interesting insight
:::

::: {.cell .code execution_count="152" execution="{\"iopub.execute_input\":\"2023-01-15T03:12:08.396913Z\",\"iopub.status.busy\":\"2023-01-15T03:12:08.396503Z\",\"iopub.status.idle\":\"2023-01-15T03:12:08.650564Z\",\"shell.execute_reply\":\"2023-01-15T03:12:08.649270Z\",\"shell.execute_reply.started\":\"2023-01-15T03:12:08.396877Z\"}" trusted="true"}
``` python
df.plot.scatter(x = 'Radius(R/Ro)', y = 'Star type')
```

::: {.output .execute_result execution_count="152"}
    <AxesSubplot:xlabel='Radius(R/Ro)', ylabel='Star type'>
:::

::: {.output .display_data}
![](vertopal_2eb67dc2b1c046c6ae9007e8cc5ee010/cac45c109020089e42ea1ce9819ae09b7e4729f5.png)
:::
:::

::: {.cell .markdown}
# Another strong emergent pattern appears in Absolute Magnitude vs Star Type, I suspect this would make a ML model\'s accuracy greatly helped with this data if used to test real world star data (which we are), due to the neatness of the data
:::

::: {.cell .code execution_count="153" execution="{\"iopub.execute_input\":\"2023-01-15T03:12:08.653076Z\",\"iopub.status.busy\":\"2023-01-15T03:12:08.652642Z\",\"iopub.status.idle\":\"2023-01-15T03:12:08.904678Z\",\"shell.execute_reply\":\"2023-01-15T03:12:08.903327Z\",\"shell.execute_reply.started\":\"2023-01-15T03:12:08.653037Z\"}" trusted="true"}
``` python
df.plot.scatter(x = 'Absolute magnitude(Mv)', y = 'Star type')
```

::: {.output .execute_result execution_count="153"}
    <AxesSubplot:xlabel='Absolute magnitude(Mv)', ylabel='Star type'>
:::

::: {.output .display_data}
![](vertopal_2eb67dc2b1c046c6ae9007e8cc5ee010/6dc276df5e691b88373f9e49733b11e0e24be013.png)
:::
:::

::: {.cell .code execution_count="154" execution="{\"iopub.execute_input\":\"2023-01-15T03:12:08.907601Z\",\"iopub.status.busy\":\"2023-01-15T03:12:08.906812Z\",\"iopub.status.idle\":\"2023-01-15T03:12:09.257177Z\",\"shell.execute_reply\":\"2023-01-15T03:12:09.255586Z\",\"shell.execute_reply.started\":\"2023-01-15T03:12:08.907551Z\"}" trusted="true"}
``` python
df.plot.scatter(x = 'Star color', y = 'Temperature (K)', figsize=(20, 4))
```

::: {.output .execute_result execution_count="154"}
    <AxesSubplot:xlabel='Star color', ylabel='Temperature (K)'>
:::

::: {.output .display_data}
![](vertopal_2eb67dc2b1c046c6ae9007e8cc5ee010/caf4d1e95996d524b01f03c1de156dc8aca87c0a.png)
:::
:::

::: {.cell .markdown}
# After seeing these relationships, its time to see how our data fits in with the Hertzsprung-Russell Diagram

`<img src="https://chandra.harvard.edu/graphics/edu/formal/variable_stars/HR_diagram.jpg" alt="HR Diagram" width="500"/>`{=html}
:::

::: {.cell .code execution_count="175" execution="{\"iopub.execute_input\":\"2023-01-15T04:08:10.591887Z\",\"iopub.status.busy\":\"2023-01-15T04:08:10.591492Z\",\"iopub.status.idle\":\"2023-01-15T04:08:10.972651Z\",\"shell.execute_reply\":\"2023-01-15T04:08:10.971378Z\",\"shell.execute_reply.started\":\"2023-01-15T04:08:10.591855Z\"}" trusted="true"}
``` python
HRGraph = df.plot.scatter(x = 'Temperature (K)', y = 'Absolute magnitude(Mv)', figsize=(6, 6))
HRGraph.set_xscale('log')
HRGraph.invert_xaxis()
HRGraph.invert_yaxis()
```

::: {.output .display_data}
![](vertopal_2eb67dc2b1c046c6ae9007e8cc5ee010/9c8e2b267ec1ac9f78d4ef7d17ea6f29e3c793e3.png)
:::
:::

::: {.cell .markdown}
# \^ It fits in moderately well, white dwarfs, giants, and main sequence patterns can be distinctly made out {#-it-fits-in-moderately-well-white-dwarfs-giants-and-main-sequence-patterns-can-be-distinctly-made-out}
:::

::: {.cell .markdown}
# Before passing in values to a ML model, let\'s see our categorical variables - the model cannot interpret those
:::

::: {.cell .code execution_count="156" execution="{\"iopub.execute_input\":\"2023-01-15T03:12:09.682198Z\",\"iopub.status.busy\":\"2023-01-15T03:12:09.681741Z\",\"iopub.status.idle\":\"2023-01-15T03:12:09.692803Z\",\"shell.execute_reply\":\"2023-01-15T03:12:09.691234Z\",\"shell.execute_reply.started\":\"2023-01-15T03:12:09.682145Z\"}" trusted="true"}
``` python
df.dtypes
```

::: {.output .execute_result execution_count="156"}
    Temperature (K)             int64
    Luminosity(L/Lo)          float64
    Radius(R/Ro)              float64
    Absolute magnitude(Mv)    float64
    Star type                   int64
    Star color                 object
    Spectral Class             object
    dtype: object
:::
:::

::: {.cell .markdown}
# Spectral Class and Star Color are 2 categorical variables, we can see these below
:::

::: {.cell .code execution_count="157" execution="{\"iopub.execute_input\":\"2023-01-15T03:12:09.696910Z\",\"iopub.status.busy\":\"2023-01-15T03:12:09.696119Z\",\"iopub.status.idle\":\"2023-01-15T03:12:09.707856Z\",\"shell.execute_reply\":\"2023-01-15T03:12:09.706230Z\",\"shell.execute_reply.started\":\"2023-01-15T03:12:09.696857Z\"}" trusted="true"}
``` python
cat_col= [col for col in df.columns if df[col].dtype == 'object']

for col in cat_col:
    print('{} has {} values '.format(col,df[col].unique()) + '\n')
```

::: {.output .stream .stdout}
    Star color has ['Red' 'Blue White' 'White' 'Yellowish White' 'Blue white'
     'Pale yellow orange' 'Blue' 'Blue-white' 'Whitish' 'yellow-white'
     'Orange' 'White-Yellow' 'white' 'Blue ' 'yellowish' 'Yellowish'
     'Orange-Red' 'Blue white ' 'Blue-White'] values 

    Spectral Class has ['M' 'B' 'A' 'F' 'O' 'K' 'G'] values 
:::
:::

::: {.cell .markdown}
# Let\'s convert these to numerical values using LabelEncoder module
:::

::: {.cell .code execution_count="171" execution="{\"iopub.execute_input\":\"2023-01-15T03:27:05.925972Z\",\"iopub.status.busy\":\"2023-01-15T03:27:05.925477Z\",\"iopub.status.idle\":\"2023-01-15T03:27:05.933948Z\",\"shell.execute_reply\":\"2023-01-15T03:27:05.932355Z\",\"shell.execute_reply.started\":\"2023-01-15T03:27:05.925911Z\"}" trusted="true"}
``` python
from sklearn.preprocessing import LabelEncoder

for col in cat_col:
    df[col] = LabelEncoder().fit_transform(df[col])
```
:::

::: {.cell .markdown}
# Let\'s confirm what our transformed categorical variables look like
:::

::: {.cell .code execution_count="159" execution="{\"iopub.execute_input\":\"2023-01-15T03:12:09.724986Z\",\"iopub.status.busy\":\"2023-01-15T03:12:09.724594Z\",\"iopub.status.idle\":\"2023-01-15T03:12:09.736929Z\",\"shell.execute_reply\":\"2023-01-15T03:12:09.735557Z\",\"shell.execute_reply.started\":\"2023-01-15T03:12:09.724952Z\"}" trusted="true"}
``` python
for col in cat_col:
    print('{} has {} values'.format(col,df[col].unique()) + '\n')
```

::: {.output .stream .stdout}
    Star color has [10  2 11 15  3  9  0  6 13 17  7 12 16  1 18 14  8  4  5] values

    Spectral Class has [5 1 0 2 6 4 3] values
:::
:::

::: {.cell .markdown}
# Discrete integers can now be processed onto the ML model, we verify by looking at all types now
:::

::: {.cell .code execution_count="160" execution="{\"iopub.execute_input\":\"2023-01-15T03:12:09.740557Z\",\"iopub.status.busy\":\"2023-01-15T03:12:09.738656Z\",\"iopub.status.idle\":\"2023-01-15T03:12:09.752706Z\",\"shell.execute_reply\":\"2023-01-15T03:12:09.751257Z\",\"shell.execute_reply.started\":\"2023-01-15T03:12:09.740501Z\"}" trusted="true"}
``` python
df.dtypes
```

::: {.output .execute_result execution_count="160"}
    Temperature (K)             int64
    Luminosity(L/Lo)          float64
    Radius(R/Ro)              float64
    Absolute magnitude(Mv)    float64
    Star type                   int64
    Star color                  int64
    Spectral Class              int64
    dtype: object
:::
:::

::: {.cell .markdown}
# We now create independent and dependent variables, i.e. feature and target variables. Star type will be target {#we-now-create-independent-and-dependent-variables-ie-feature-and-target-variables-star-type-will-be-target}
:::

::: {.cell .code execution_count="161" execution="{\"iopub.execute_input\":\"2023-01-15T03:12:09.755701Z\",\"iopub.status.busy\":\"2023-01-15T03:12:09.754769Z\",\"iopub.status.idle\":\"2023-01-15T03:12:09.775579Z\",\"shell.execute_reply\":\"2023-01-15T03:12:09.774143Z\",\"shell.execute_reply.started\":\"2023-01-15T03:12:09.755660Z\"}" trusted="true"}
``` python
ind_col = [col for col in df.columns if col!='Star type']
dep_col = 'Star type'
X = df[ind_col]
Y = df[dep_col]
X.head()
```

::: {.output .execute_result execution_count="161"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Temperature (K)</th>
      <th>Luminosity(L/Lo)</th>
      <th>Radius(R/Ro)</th>
      <th>Absolute magnitude(Mv)</th>
      <th>Star color</th>
      <th>Spectral Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3068</td>
      <td>0.002400</td>
      <td>0.1700</td>
      <td>16.12</td>
      <td>10</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3042</td>
      <td>0.000500</td>
      <td>0.1542</td>
      <td>16.60</td>
      <td>10</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2600</td>
      <td>0.000300</td>
      <td>0.1020</td>
      <td>18.70</td>
      <td>10</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2800</td>
      <td>0.000200</td>
      <td>0.1600</td>
      <td>16.65</td>
      <td>10</td>
      <td>5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1939</td>
      <td>0.000138</td>
      <td>0.1030</td>
      <td>20.06</td>
      <td>10</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {.cell .markdown}
# Data dimensions for both total datasets are below
:::

::: {.cell .code execution_count="162" execution="{\"iopub.execute_input\":\"2023-01-15T03:12:09.778141Z\",\"iopub.status.busy\":\"2023-01-15T03:12:09.777751Z\",\"iopub.status.idle\":\"2023-01-15T03:12:09.785708Z\",\"shell.execute_reply\":\"2023-01-15T03:12:09.784747Z\",\"shell.execute_reply.started\":\"2023-01-15T03:12:09.778106Z\"}" trusted="true"}
``` python
print('X Shape:' + str(X.shape) + '| Y Shape: ' + str(Y.shape))
```

::: {.output .stream .stdout}
    X Shape:(240, 6)| Y Shape: (240,)
:::
:::

::: {.cell .markdown}
# Let\'s split our data into training the model and testing model. A common an effective ratio usually is 80/20 so we\'ll do that {#lets-split-our-data-into-training-the-model-and-testing-model-a-common-an-effective-ratio-usually-is-8020-so-well-do-that}
:::

::: {.cell .code execution_count="163" execution="{\"iopub.execute_input\":\"2023-01-15T03:12:09.787780Z\",\"iopub.status.busy\":\"2023-01-15T03:12:09.787072Z\",\"iopub.status.idle\":\"2023-01-15T03:12:09.800778Z\",\"shell.execute_reply\":\"2023-01-15T03:12:09.799579Z\",\"shell.execute_reply.started\":\"2023-01-15T03:12:09.787745Z\"}" trusted="true"}
``` python
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,random_state=0, test_size=0.2)
print('X Shape:' + str(X_train.shape) + '| Y Shape: ' + str(Y_test.shape))
```

::: {.output .stream .stdout}
    X Shape:(192, 6)| Y Shape: (48,)
:::
:::

::: {.cell .markdown}
# Now we import the ML module to generate a model - XGBoost is a strong library for this
:::

::: {.cell .code execution_count="164" execution="{\"iopub.execute_input\":\"2023-01-15T03:12:09.803071Z\",\"iopub.status.busy\":\"2023-01-15T03:12:09.802180Z\",\"iopub.status.idle\":\"2023-01-15T03:12:09.810822Z\",\"shell.execute_reply\":\"2023-01-15T03:12:09.809745Z\",\"shell.execute_reply.started\":\"2023-01-15T03:12:09.803015Z\"}" trusted="true"}
``` python
from xgboost import XGBClassifier
```
:::

::: {.cell .markdown}
# We also want cross-validation as it can make the ML model more reliable in predicting new sets of data by gaining training exposure to different subsets in the training set
:::

::: {.cell .code execution_count="165" execution="{\"iopub.execute_input\":\"2023-01-15T03:12:09.813251Z\",\"iopub.status.busy\":\"2023-01-15T03:12:09.812129Z\",\"iopub.status.idle\":\"2023-01-15T03:12:13.010895Z\",\"shell.execute_reply\":\"2023-01-15T03:12:13.009732Z\",\"shell.execute_reply.started\":\"2023-01-15T03:12:09.813206Z\"}" trusted="true"}
``` python
classifier = XGBClassifier()

params = {
 'learning_rate' : [0.05,0.10,0.15,0.20,0.25,0.30],
 'max_depth' : [ 3, 4, 5, 6, 8, 10, 12, 15],
 'min_child_weight' : [ 1, 3, 5, 7 ],
 'gamma': [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
 'colsample_bytree' : [ 0.3, 0.4, 0.5 , 0.7 ]
}

from sklearn.model_selection import RandomizedSearchCV

rs_cv=RandomizedSearchCV(classifier,param_distributions=params,n_iter=5,scoring='roc_auc',n_jobs=-1,cv=5,verbose=3)
rs_cv.fit(X_train, Y_train)

rs_cv.best_estimator_
```

::: {.output .stream .stdout}
    Fitting 5 folds for each of 5 candidates, totalling 25 fits
:::

::: {.output .stream .stderr}
    /opt/conda/lib/python3.7/site-packages/sklearn/model_selection/_search.py:972: UserWarning: One or more of the test scores are non-finite: [nan nan nan nan nan]
      category=UserWarning,
:::

::: {.output .execute_result execution_count="165"}
    XGBClassifier(base_score=0.5, booster='gbtree', callbacks=None,
                  colsample_bylevel=1, colsample_bynode=1, colsample_bytree=0.4,
                  early_stopping_rounds=None, enable_categorical=False,
                  eval_metric=None, gamma=0.1, gpu_id=-1, grow_policy='depthwise',
                  importance_type=None, interaction_constraints='',
                  learning_rate=0.05, max_bin=256, max_cat_to_onehot=4,
                  max_delta_step=0, max_depth=12, max_leaves=0, min_child_weight=7,
                  missing=nan, monotone_constraints='()', n_estimators=100,
                  n_jobs=0, num_parallel_tree=1, objective='multi:softprob',
                  predictor='auto', random_state=0, reg_alpha=0, ...)
:::
:::

::: {.cell .markdown}
# The RandomizedSearchCV library gives us the optimal parameters for the Classifier Model, now we have a working model to test
:::

::: {.cell .code execution_count="166" execution="{\"iopub.execute_input\":\"2023-01-15T03:12:13.012797Z\",\"iopub.status.busy\":\"2023-01-15T03:12:13.012437Z\",\"iopub.status.idle\":\"2023-01-15T03:12:13.343744Z\",\"shell.execute_reply\":\"2023-01-15T03:12:13.342842Z\",\"shell.execute_reply.started\":\"2023-01-15T03:12:13.012764Z\"}" trusted="true"}
``` python
classifier = XGBClassifier(base_score=0.5, booster='gbtree', callbacks=None,
              colsample_bylevel=1, colsample_bynode=1, colsample_bytree=0.4,
              early_stopping_rounds=None, enable_categorical=False,
              eval_metric=None, gamma=0.1, gpu_id=-1, grow_policy='depthwise',
              importance_type=None, interaction_constraints='',
              learning_rate=0.05, max_bin=256, max_cat_to_onehot=4,
              max_delta_step=0, max_depth=12, max_leaves=0, min_child_weight=7,
              monotone_constraints='()', n_estimators=100,
              n_jobs=0, num_parallel_tree=1, objective='multi:softprob',
              predictor='auto', random_state=0, reg_alpha=0)
classifier.fit(X_train, Y_train)
```

::: {.output .execute_result execution_count="166"}
    XGBClassifier(base_score=0.5, booster='gbtree', callbacks=None,
                  colsample_bylevel=1, colsample_bynode=1, colsample_bytree=0.7,
                  early_stopping_rounds=None, enable_categorical=False,
                  eval_metric=None, gamma=0.0, gpu_id=-1, grow_policy='depthwise',
                  importance_type=None, interaction_constraints='',
                  learning_rate=0.1, max_bin=256, max_cat_to_onehot=4,
                  max_delta_step=0, max_depth=12, max_leaves=0, min_child_weight=5,
                  missing=nan, monotone_constraints='()', n_estimators=100,
                  n_jobs=0, num_parallel_tree=1, objective='multi:softprob',
                  predictor='auto', random_state=0, reg_alpha=0, ...)
:::
:::

::: {.cell .markdown}
# Let\'s see how our model did in predicting star types
:::

::: {.cell .code execution_count="172" execution="{\"iopub.execute_input\":\"2023-01-15T03:39:10.987600Z\",\"iopub.status.busy\":\"2023-01-15T03:39:10.987141Z\",\"iopub.status.idle\":\"2023-01-15T03:39:11.009548Z\",\"shell.execute_reply\":\"2023-01-15T03:39:11.007387Z\",\"shell.execute_reply.started\":\"2023-01-15T03:39:10.987534Z\"}" trusted="true"}
``` python
Y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score

accuracy_score(Y_test, Y_pred)
```

::: {.output .execute_result execution_count="172"}
    1.0
:::
:::

::: {.cell .markdown}
# We got a 100% accuracy rate in predicting star types! That\'s a very neat model
:::

::: {.cell .code execution_count="173" execution="{\"iopub.execute_input\":\"2023-01-15T03:39:24.166609Z\",\"iopub.status.busy\":\"2023-01-15T03:39:24.166097Z\",\"iopub.status.idle\":\"2023-01-15T03:39:24.177171Z\",\"shell.execute_reply\":\"2023-01-15T03:39:24.175799Z\",\"shell.execute_reply.started\":\"2023-01-15T03:39:24.166569Z\"}" trusted="true"}
``` python
cm = confusion_matrix(Y_test, Y_pred)
cm
```

::: {.output .execute_result execution_count="173"}
    array([[ 7,  0,  0,  0,  0,  0],
           [ 0,  9,  0,  0,  0,  0],
           [ 0,  0,  7,  0,  0,  0],
           [ 0,  0,  0,  8,  0,  0],
           [ 0,  0,  0,  0, 11,  0],
           [ 0,  0,  0,  0,  0,  6]])
:::
:::

::: {.cell .markdown}
# We see the breakdown of how many and which stars it categorized into its respective type

# These numbers make sense as if you sum them, you get the number of values in Y_test - our testing dataset!
:::

::: {.cell .code execution_count="174" execution="{\"iopub.execute_input\":\"2023-01-15T03:39:29.520090Z\",\"iopub.status.busy\":\"2023-01-15T03:39:29.519612Z\",\"iopub.status.idle\":\"2023-01-15T03:39:29.526544Z\",\"shell.execute_reply\":\"2023-01-15T03:39:29.525575Z\",\"shell.execute_reply.started\":\"2023-01-15T03:39:29.520048Z\"}" trusted="true"}
``` python
sum = 0
for x in cm:
    for y in x:
        sum+=y
        
print(sum)
```

::: {.output .stream .stdout}
    48
:::
:::
