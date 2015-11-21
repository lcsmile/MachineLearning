### Machine Learning Project Writeup

This file is the final write-up for the John Hopkins University Coursera course ‘Practical Machine Learning’. 
Course link: [https://www.coursera.org/course/predmachlearn]


## Answer to questions:
# How the model was built: 
A Decision Tree model and a Random Forest model are built to predict how people do exercise using devices, after preliminary analysis of the problem and data preprocessing.
Final model (the Random Forest tree model) will be selected based on the model performance in terms of accuracy and out-of-sample error. 

# Cross-validation:
The provided training datasets are partitioned into a sub-training set and a sub-validation set based on 0.75/0.25 split. 10-fold cross validation is applied during model fitting.

# Expected out-of-sample error:
According to the definition, out-of-sample error is the error rate on a new dataset. It is critical in terms of measuring the performance of model. It is used to estimate the model based on data up to and including today. Mathematically, we could use the difference between 1 and the accuracy of the model on validation set. 
In this case, our out-of-sample error is 25.92% and 0.65% for the Decision Tree and Random Forest model respectively. This indicates the better performance of the Random Forest model.

# Reasons for my choices:
Since we have a large size of data, we divide the given training dataset into a sub-training and a sub-validation set to measure the performance of the model.
Our goal is to analyze and predict variable ‘classe’ based on the rest of given variables, which is a categorical variable with 5 levels. Therefore we do not expect the rest of variables are linear or interacting in a linear way. In this case, Random Forests and Decision Tree models are good candidates over other models like linear models. In addition, these models are constructed to deal with high-dimension and large size of data. 


**_All work is implemented with R._**

**Background** (quoted from the course assignment website):  
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: [http://groupware.les.inf.puc-rio.br/har] (see the section on the Weight Lifting Exercise Dataset). 

**Data source**
Data is provided from [http://groupware.les.inf.puc-rio.br/har]. Training and Testing datasets are downloaded with the following R commands. There are 160 columns and 19622 rows in the original training dataset, 160 columns (exactly the same names) and 20 rows in testing dataset.

```
#download data to R working directory, and read data
#note that there are cells with missing, 'NA' or '#DIV/0!' values. We changed them to be all ‘NA’ values
#training set                                                                                                                                       
training=read.csv('E:/Couresa-write up/pml-training.csv',na.string=c("", "NA", "#DIV/0!"))
#testing set
testing=read.csv('E:/Couresa-write up/pml-testing.csv',header=T,na.string=c("", "NA", "#DIV/0!"))
#check dimensions 
dim(training)
##[1] 19622   160
dim(testing)
##[1]  20 160
table(training$classe)
##   A    B    C    D    E 
##5580 3797 3422 3216 3607
```

# Data Preprocessing
**Data Cleaning**
After taking a look at variable names and values in the provided datasets, we noticed that 7 columns (variables) are not necessarily related to ‘classe’. Specifically, column ‘X’ is an index variable; column ‘user_name’, ‘raw_timestamp_part_1’, ‘raw_timestamp_part_2’,’cvtd_timestamp’, ‘new_window’,’num_window’ are not directly contributing to the values of ‘classe’. Therefore we are going to remove these 7 variables.
At the same time, we are going to remove columns with ‘NA’ values. 
Now there are 53 columns/variables remaining in both training and testing datasets.
A bar chart of different levels of ‘classe’ variables below shows the count of each levels of variable ‘classe’. 

![Classe Level Bar Chart](https://github.com/lcsmile/MachineLearning/figures/classe_level.png)

```
names(training_new)
#remove columns with NA values
training_new=training[,!apply(training,2,function(training)any(is.na(training)))]
testing_new=testing[,!apply(testing,2,function(testing)any(is.na(testing)))]

#remove first 7 columns, they are unrelated to classe
training_new=training_new[,-c(1:7)]
testing_new=testing_new[,-c(1:7)]
dim(training_new)
##[1] 19622    53
dim(testing_new) 
###[1] 20 53
```

**Data Slicing**
Before fitting a model with the provided data, we have to split the given training datasets in to a new ‘training set’ (75%) and a ‘validation set’ (25%) without replacement. So the new training set contains 14718 records and new validation set contains 4904 records.

```
#partitioning the training set: 75% vs 25%
subTrain=createDataPartition(y=training_new$classe,p=0.75,list=FALSE)
subTraining=training_new[subTrain,]
subTesting=training_new[-subTrain,]
subTraining=subTraining[,!apply(subTraining,2,function(subTraining)any(is.na(subTraining)))]
dim(subTraining)
##[1] 14718    53
dim(subTesting)
##[1] 4904   53
```

# Data Modeling
First we are to download the required packages:

```
#download and install packages
install.packages('caret')
install.packages('rpart')
install.packages('randomForest')

#access packages
library(caret)
library(ggplot2)
library(randomForest)
library(rpart)

#set seed
set.seed(1116)
```

**Model1: Classification Tree**
First we are going to fit the data with a Classification Tree model. A 10-fold cross validation is applied. The result of a tree plot is shown as below. 


![Classe Level Bar Chart](https://github.com/lcsmile/MachineLearning/figures/classification_tree.png)

```
#decision tree
fit1=rpart(classe~.,data=subTraining,control=rpart.control(xval=10,minbucket=2,cp=0.01),method='class')
#plot decision tree
library(rpart.plot)
rpart.plot(fit1)
plot(fit1,main='classification tree')
text(fit1)
```

Now we are going to measure the performance of this model by predicting the validation set. From the result of following R command, the accuracy is 74.08% and out-of-sample error is around 25.92%.

```
#prediction with decision tree
pred1=predict(fit1,subTesting,type='class')
#accuracy 
#confusion matrix
confusionMatrix(pred1,subTesting$classe)

##Confusion Matrix and Statistics

##          Reference
##Prediction    A    B    C    D    E
##         A 1282  204   25   86   34
##         B   43  510   48   25   56
##         C   34   96  656  113   91
##         D   13   62   58  510   45
##         E   23   77   68   70  675

##Overall Statistics
                                         
##               Accuracy : 0.7408         
##                 95% CI : (0.7283, 0.753)
##    No Information Rate : 0.2845         
##    P-Value [Acc > NIR] : < 2.2e-16      
                                         
##                  Kappa : 0.6703         
## Mcnemar's Test P-Value : < 2.2e-16      

##Statistics by Class:

##                     Class: A Class: B Class: C Class: D Class: E
##Sensitivity            0.9190   0.5374   0.7673   0.6343   0.7492
##Specificity            0.9005   0.9565   0.9175   0.9566   0.9405
##Pos Pred Value         0.7860   0.7478   0.6626   0.7413   0.7393
##Neg Pred Value         0.9655   0.8960   0.9492   0.9303   0.9434
##Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
##Detection Rate         0.2614   0.1040   0.1338   0.1040   0.1376
##Detection Prevalence   0.3326   0.1391   0.2019   0.1403   0.1862
##Balanced Accuracy      0.9098   0.7470   0.8424   0.7955   0.8449
##out_error_fit1=1-as.numeric(confusionMatrix(subTesting$classe,pred1)$overall[1])
##out_error_fit1
##[1] 0.2591762
```

**Model2: Random Forest:** 
The second model is using Random Forest algorithm. A 10-fold cross validation is applied.

```
control.fit2=trainControl(method='cv',10)
fit2=train(classe~.,data=subTraining,method='rf',trControl=control.fit2,ntree=200)
fit2

##Random Forest 
##
##14718 samples
##   52 predictor
##    5 classes: 'A', 'B', 'C', 'D', 'E' 
##
##No pre-processing
##Resampling: Cross-Validated (10 fold) 
##Summary of sample sizes: 13245, 13245, 13247, 13246, 13247, 13246, ... 
##Resampling results across tuning parameters:
##
## mtry  Accuracy   Kappa      Accuracy SD  Kappa SD   
##  2    0.9908961  0.9884824  0.002356305  0.002981588
## 27    0.9918473  0.9896866  0.001429279  0.001807898
##52    0.9855288  0.9816933  0.002877971  0.003639458
##
##Accuracy was used to select the optimal model using  the largest value.
##The final value used for the model was mtry = 27.
```

Now we measure the performance of this model by predicting on the validation set. From the result of commands below, we can tell that the accuracy of this model is 99.35% and out-of-sample error is around 0.65%.

```
#prediction with Random Forest model
pred2=predict(fit2,subTesting)

#accuracy
confusionMatrix(pred2,subTesting$classe)

##Confusion Matrix and Statistics

##          Reference
##Prediction    A    B    C    D    E
##         A 1393   12    0    0    0
##         B    1  933    2    0    0
##         C    1    3  851    7    1
##         D    0    1    2  797    2
##         E    0    0    0    0  898

##Overall Statistics
                                          
##               Accuracy : 0.9935          
##                 95% CI : (0.9908, 0.9955)
##    No Information Rate : 0.2845          
##    P-Value [Acc > NIR] : < 2.2e-16       
                                          
##                  Kappa : 0.9917          
## Mcnemar's Test P-Value : NA              

##Statistics by Class:

##                     Class: A Class: B Class: C Class: D Class: E
##Sensitivity            0.9986   0.9831   0.9953   0.9913   0.9967
##Specificity            0.9966   0.9992   0.9970   0.9988   1.0000
##Pos Pred Value         0.9915   0.9968   0.9861   0.9938   1.0000
##Neg Pred Value         0.9994   0.9960   0.9990   0.9983   0.9993
##Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
##Detection Rate         0.2841   0.1903   0.1735   0.1625   0.1831
##Detection Prevalence   0.2865   0.1909   0.1760   0.1635   0.1831
##Balanced Accuracy      0.9976   0.9912   0.9962   0.9950   0.9983

out_error_fit2=1-as.numeric(confusionMatrix(subTesting$classe,pred2)$overall[1])
out_error_fit2
##[1] 0.006525285
```

**Model Selection and Predicting for Test Dataset**
From the above result, we are going to choose the Random Forest model since it has a higher accuracy and apply it to the test set provided. 
Note that we already preprocessed the test set in the same way as we did for training set at the beginning. 

```
###Prediction on the unknown sample
predictTest_rf=predict(fit2,testing_new)
predictTest_rf

##[1] B A B A A E D B A A B C B A E E A B B B
##Levels: A B C D E

#Write files for submission:
#> write in files
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}
pml_write_files(as.character(predictTest_rf))
```
