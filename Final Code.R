########################### libraries ###############################

library(ggplot2)
library(dendextend)
library(caret)
library(MASS)
library(DataExplorer)
library(class)
library(ISLR)
library(caret)
library(lattice)
library(nnet)
library(randomForest)
library(microbenchmark)
install.packages('e1071')
library(e1071) 

########################### Data extraction ###############################


train=read.csv("/Users/vaishnavipatki/Downloads/Project 2/Trial/train.csv",header = TRUE,sep =",")
test=read.csv("/Users/vaishnavipatki/Downloads/Project 2/Trial/test.csv",header = TRUE,sep=",")

summary(train)
attach(train)
head(train)
sum(is.null(train))
sum(is.null(test))

########################### Data combining ###############################

mydata=rbind(train,test)
train.par=train
train.par$partition="train"
train.par=transform(train.par,sID=factor(subject),aID=factor(Activity))

test.par=test
test.par$partition="test"
test.par=transform(test.par,sID=factor(subject),aID=factor(Activity))

mydata.partition=rbind(train.par,test.par)

######################### Data Vizualization ############################

qplot(data = mydata.partition,x=sID,fill=aID)
qplot(data = mydata.partition,x=sID,fill=partition )
plot_missing(train)
plot_bar(train)
plot_correlation(train)

###################### Splitting the data ########################
set.seed(1)
train.x <- train[, -c(562:563)]
train.y <- train[,"Activity"]
test.x <- test[, -c(562:563)]
test.y <- test[,"Activity"]

###################### Applying PCA on train #############################
set.seed(1)
pc.x <- prcomp (train.x , scale =TRUE)
pr.var =pc.x$sdev ^2
pve=pr.var/sum(pr.var)
plot(pve , xlab="Principal Component", ylab="Proportion of Variance Explained", ylim=c(0,1) ,type="b")
plot(cumsum (pve), xlab=" Principal Component ", ylab ="Cumulative Proportion of Variance Explained ", ylim=c(0,1) , type="b")


cumsum (pve)*100

train.data <- data.frame(pc.x$x[, 1:102])
train.pca = data.frame(pc.x$x[, 1:102], train.y)


############### PCA on test ################# 
set.seed(1)
test.data <- predict(pc.x, newdata=test.x)
test.data <- as.data.frame(test.data)
test.data <- test.data[,1:102]

test.pca= data.frame(test.data, test.y)



######### Combining test and train data frames after PCA#####

my.data = rbind(train.data,test.data)

######################### Hierarchical cluster ######################

#we perform hierachical clustering on subject 3 for vizualization 

subject.3=subset(mydata,subject==3)
distances=dist(subject.3[,-c(562:563)])
hcluster=hclust(distances,method = "complete")
dendr=as.dendrogram(hcluster)
dendr=color_branches(dendr,k=6)
plot(dendr)


####################### K means #########################

k.out= kmeans(c(my.data[,1],my.data[,2]), 6,nstart=25)
kcluster=k.out$cluster
plot(my.data[,1],my.data[,2],col=kcluster,xlab='PC1',ylab='PC2')
legend("topleft",legend=unique(train.pca$train.y),col=unique(train.pca$train.y),cex=0.6,pch=1)


#################### Logistics regression ######################
t1<-Sys.time()
set.seed(1)
fit.mlr = multinom(train.y~., data =train.pca)
fit.mlr
pred.mlr = predict(fit.mlr,test.pca)
CM = table(pred.mlr,test.pca$test.y)
CM
acc = (sum(diag(CM)))/sum(CM)
acc
t2<-Sys.time()
print(t2-t1)

Logistic_function<-function(train.y,train.pca,test.pca){
  set.seed(1)
  fit.mlr = multinom(train.y~., data =train.pca)
  fit.mlr
  pred.mlr = predict(fit.mlr,test.pca)
  CM = table(pred.mlr,test.pca$test.y)
  CM
  acc = (sum(diag(CM)))/sum(CM)
  acc
}

############## LDA #############
LDA_function<-function(train.y,train.pca,test.pca){
  set.seed(1)
  lda1<-lda(train.y~., data = train.pca)
  lda1
  pred<-predict(lda1,test.pca)
  CM = table(pred$class,test.pca$test.y)
  CM
  acc = (sum(diag(CM)))/sum(CM)
  acc
}
t1<-Sys.time()
set.seed(1)
lda1<-lda(train.y~., data = train.pca)
lda1
pred<-predict(lda1,test.pca)
CM = table(pred$class,test.pca$test.y)
CM
acc = (sum(diag(CM)))/sum(CM)
acc
t2<-Sys.time()
print(t2-t1)
############ QDA ##############
QDA_function<-function(train.y,train.pca,test.pca){
  set.seed(1)
  qda1<-qda(train.y~., data =train.pca)
  qda1
  pred<-predict(qda1,test.data)
  CM = table(pred$class,test.pca$test.y)
  CM
  acc = (sum(diag(CM)))/sum(CM)
  acc

}
t1<-Sys.time()
set.seed(1)
qda1<-qda(train.y~., data =train.pca)
qda1
pred<-predict(qda1,test.data)
CM = table(pred$class,test.pca$test.y)
CM
acc = (sum(diag(CM)))/sum(CM)
acc
t2<-Sys.time()
print(t2-t1)
#################### KNN #######################

## K=5 #########
set.seed(1)
knn.fit5=knn(train.data,test.data,train.y,k=5)
summary(knn.fit5)
CM = table(knn.fit5, test.y)
acc = (sum(diag(CM)))/sum(CM)
acc
### K=10 ########
t1<-Sys.time()
set.seed(1)
knn.fit10=knn(train.data,test.data,train.y,k=10)
summary(knn.fit10)
CM = table(knn.fit10, test.y)
acc = (sum(diag(CM)))/sum(CM)
acc
t2<-Sys.time()
print(t2-t1)
### K=50##########
set.seed(1)
knn.fit50=knn(train.data,test.data,train.y,k=50)
summary(knn.fit10)
CM = table(knn.fit50, test.y)
acc = (sum(diag(CM)))/sum(CM)
acc

##### K=100 #########
set.seed(1)
knn.fit100=knn(train.data,test.data,train.y,k=100)
summary(knn.fit100)
CM = table(knn.fit100, test.y)
acc = (sum(diag(CM)))/sum(CM)
acc

##### K=25 #########
set.seed(1)
knn.fit100=knn(train.data,test.data,train.y,k=25)
summary(knn.fit100)
CM = table(knn.fit100, test.y)
acc = (sum(diag(CM)))/sum(CM)
acc

KNN_function<-function(train.data,test.data,train.y){
  set.seed(1)
  knn.fit10=knn(train.data,test.data,train.y,k=10)
  summary(knn.fit10)
  CM = table(knn.fit10, test.y)
  acc = (sum(diag(CM)))/sum(CM)
  acc
  
}

################### Random Forests ######################
t1<-Sys.time()
set.seed(1)
rf <- round((102)^0.5)
fit.rf = randomForest(factor(train.y) ~ ., data=train.pca, mtry = rf, ntree=300, importance = TRUE)
fitrf.pred <-predict(fit.rf, newdata=test.data)
Accuracy <- mean(fitrf.pred == test.y)
Accuracy
t2<-Sys.time()
print(t2-t1)
importance(fit.rf)
varImpPlot(fit.rf)

RF_function<-function(train.y,test.y,train.pca,test.data){
  rf <- round((102)^0.5)
  fit.rf = randomForest(factor(train.y) ~ ., data=train.pca, mtry = rf, ntree=300, importance = TRUE)
  fitrf.pred <-predict(fit.rf, newdata=test.data)
  Accuracy <- mean(fitrf.pred == test.y)
  Accuracy 
}


################## Bagging ###################
t1<-Sys.time()
bag.fit.tree <- randomForest(factor(train.y) ~., data=train.pca , mtry = 10, ntree = 500, importance = TRUE)
pred.bag <- predict(bag.fit.tree, newdata=test.data)
Accuracy <- mean(pred.bag == test.y)
Accuracy
t2<-Sys.time()
print(t2-t1)
importance(bag.fit.tree)
varImpPlot(bag.fit.tree)

Bagging_function<-function(train.y,test.y,train.pca,test.data){
  bag.fit.tree <- randomForest(factor(train.y) ~., data=train.pca , mtry = 10, ntree = 500, importance = TRUE) 
  pred.bag <- predict(bag.fit.tree, newdata=test.data)
  Accuracy <- mean(pred.bag == test.y)
  Accuracy
}

#########Compare the run times############
out<-microbenchmark(LDA_function(train.y,train.pca,test.pca),unit="ms")
summary(out)
out<-microbenchmark(QDA_function(train.y,train.pca,test.pca),unit="ms")
summary(out)
out<-microbenchmark(KNN_function(train.data,test.data,train.y),unit="ms")
summary(out)
#out<-microbenchmark(RF_function(train.y,test.y,train.pca,test.data),unit="ms")
#sbestummary(out)
#out<-microbenchmark(Bagging_function(train.y,test.y,train.pca,test.data),unit="ms")
#summary(out)



