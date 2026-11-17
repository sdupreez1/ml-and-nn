library('nnet')
library('faraway')
library('ggplot2')

data(ozone)
O3 = ozone$O3

# Naive training
o3.nn = nnet(O3 ~ temp + ibh + ibt, data = ozone, size = 2, linout = TRUE)

RSS = sum((o3.nn$fitted.values - O3)^2)    # can also use o3.nn$value
TSS = sum((O3 - mean(O3))^2)

RSS - TSS   # very small, so very little variance is captured by the model

oz.norm = scale(ozone)    # normalise all columns
                          # leads to better weights since numbers are all similar size
best.nn = o3.nn
RSS.best = best.nn$value 

for(i in 1:100){
  set.seed(i)   # try different seeds to minimise influence of stochastic effects
  new.nn = nnet(O3 ~ temp + ibh + ibt, data = oz.norm, size = 2, linout = TRUE)
  RSS.new = new.nn$value
  
  if(RSS.new < RSS.best){
    best.nn = new.nn
    RSS.best = new.nn$value
  }
}

TSS.norm = sum((oz.norm[,'O3'] - mean(oz.norm[,'O3']))^2)
RSS.best - TSS.norm   # much larger, new model explains more of total variance

summary(best.nn)      # check weights

# Predictions for varying temp
xt.norm = expand.grid(temp=seq(-3,3,0.1),ibh=0,ibt=0)     # predictor values (note temp vals are normalised given that nn is trained on norm data)
oz.norm.pred.temp = predict(best.nn, newdata = xt.norm)   # predicted temps for predictor values

oz.pred.temp = oz.norm.pred.temp*sd(O3) + mean(O3)        # undo normalisation
xt = xt.norm*sd(ozone$temp) + mean(ozone$temp)            # undo normalisation

# black = prediction, red = actual values
ggplot(data = xt, aes(temp, oz.pred.temp)) + 
  geom_point() + 
  geom_point(data = ozone, aes(temp, O3), col = 'red', shape = 'x')

# Predictions for varying ibh
x.ibh.norm = expand.grid(temp=0,ibh=seq(-3,3,0.1),ibt=0)
oz.norm.pred.ibh = predict(best.nn, newdata = x.ibh.norm)

oz.pred.ibh = oz.norm.pred.ibh*sd(O3) + mean(O3)
x.ibh = x.ibh.norm*sd(ozone$ibh) + mean(ozone$ibh)

ggplot(data = x.ibh, aes(ibh, oz.pred.ibh)) + 
  geom_point() + 
  geom_point(data = ozone, aes(ibh, O3), col = 'red', shape = 'x')


# Predictions for varying ibt
xibt.norm = expand.grid(temp=0,ibh=0,ibt=seq(-3,3,0.1))

oz.norm.pred.ibt = predict(best.nn, newdata = xibt.norm)

oz.pred.ibt = oz.norm.pred.ibt*sd(O3) + mean(O3)
xibt = xibt.norm*sd(ozone$ibt) + mean(ozone$ibt)

ggplot(data = xibt, aes(ibt, oz.pred.ibt)) + 
  geom_point() + 
  geom_point(data = ozone, aes(ibt, O3), col = 'red', shape = 'x')

# all three have discontinuities, possibly due to large weightings in NN
# this is handled using shrinkage methods

# Fitting an NN with shrinkage
shr.best.nn = o3.nn
shr.RSS.best = shr.best.nn$value 

for(i in 1:100){
  set.seed(i)
  shr.new.nn = nnet(O3 ~ temp + ibh +ibt, data = oz.norm,
                 size = 2, linout = TRUE, decay = 0.001)    # decay creates shrinkage
  shr.RSS.new = shr.new.nn$value
  
  if(shr.RSS.new < shr.RSS.best){  
    shr.best.nn = shr.new.nn
    shr.RSS.best = shr.new.nn$value
  }
}

shr.best.nn$value - best.nn$value   # shr has slightly larger RSS
shr.best.nn$wts - best.nn$wts       # shr has much lower weights = smoother predictions (large weights can easily lead to large jumps in predictions)

# Predictions with shrinkage
# temp
xt.norm = expand.grid(temp=seq(-3,3,0.1),ibh=0,ibt=0)
oz.norm.pred.temp = predict(shr.best.nn, newdata = xt.norm)

oz.pred.temp = oz.norm.pred.temp*sd(O3) + mean(O3)
xt = xt.norm*sd(ozone$temp) + mean(ozone$temp)

ggplot(data = xt, aes(temp, oz.pred.temp)) + 
  geom_point() + 
  geom_point(data = ozone, aes(temp, O3), col = 'red', shape = 'x')

# ibh
x.ibh.norm = expand.grid(temp=0,ibh=seq(-3,3,0.1),ibt=0)
oz.norm.pred.ibh = predict(shr.best.nn, newdata = x.ibh.norm)

oz.pred.ibh = oz.norm.pred.ibh*sd(O3) + mean(O3)
x.ibh = x.ibh.norm*sd(ozone$ibh) + mean(ozone$ibh)

ggplot(data = x.ibh, aes(ibh, oz.pred.ibh)) + 
  geom_point() + 
  geom_point(data = ozone, aes(ibh, O3), col = 'red', shape = 'x')

# ibt
xibt.norm = expand.grid(temp=0,ibh=0,ibt=seq(-3,3,0.1))
oz.norm.pred.ibt = predict(shr.best.nn, newdata = xibt.norm)

oz.pred.ibt = oz.norm.pred.ibt*sd(O3) + mean(O3)
xibt = xibt.norm*sd(ozone$ibt) + mean(ozone$ibt)

ggplot(data = xibt, aes(ibt, oz.pred.ibt)) + 
  geom_point() + 
  geom_point(data = ozone, aes(ibt, O3), col = 'red', shape = 'x')

# no discontinuities after shrinkage as expected

# NN-based classification with multiple classes in response variable
data(iris)
y = class.ind(iris$Species)   # factors -> one-hot (species)
x = iris[,1:3]                # predictors (sepal.length, sepal.width, and petal.length)

# 75 random numbers, 25 from each species (for balanced training data)
training_indexes = c(sample(1:50,25), sample(51:100,25), sample(101:150,25))

y_train = y[training_indexes,]    # response   (training)
x_train = x[training_indexes,]    # predictors (training)

y_valid = y[-training_indexes,]   # response   (validation)
x_valid = x[-training_indexes,]   # predictors (validation)

# Training a NN to predict species
x.norm = scale(x_train)
iris.nn = nnet(x = x_train, y = y_train, size = 2,
                softmax = T,      # gives log-linear model
                decay = 0.00001,
                maxit = 200)
best.iris = iris.nn
iris.RSS = best.iris$value

for(i in 1:100){
  set.seed(i)
  new.iris = nnet(x = x.norm, y = y_train, size = 2, softmax = T, 
                  decay = 0.00001,  # could use cross-validation to find optimal decay value
                  maxit = 200) 
  new.RSS = new.iris$value
  
  if(new.RSS < iris.RSS){
    best.iris = new.iris
    iris.RSS = new.RSS
  }
}

y_pred = round(predict(best.iris, newdata = scale(x_valid)))
pred_error = table(y_valid - y_pred)  # only 8 incorrect predictions from this model!
