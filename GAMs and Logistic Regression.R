library(MASS)
library(faraway)
library(splines)
library(gam)

data('seatpos')

# Polynomial regression
hip = seatpos$hipcenter                     # hip center 
ht = seatpos$Ht                             # height

l.model = lm(hip ~ ht)                      # simple linear regression
p.model = lm(hip ~ poly(ht, degree = 2))    # polynomial regression

# coefs and performance metrics
l.summ = summary(l.model)                   # linear
p.summ = summary(p.model)                   # polynomial

ht.sort = sort(ht)    # so that predictions are join in order when plotted

l.pred = predict(l.model, newdata = list(ht = ht.sort), se = TRUE)          
p.pred = predict(p.model, newdata = list(ht = ht.sort), se = TRUE)          

l.CI  = cbind(l.pred$fit - 2*l.pred$se.fit, l.pred$fit + 2*l.pred$se.fit)   
p.CI  = cbind(p.pred$fit - 2*p.pred$se.fit, p.pred$fit + 2*p.pred$se.fit)   

par(mfrow = c(1, 2))          # plot the next 2 graphs next to each other 

plot(ht, hip) + abline(l.model) + matlines(ht.sort, l.CI) 
plot(ht, hip) + lines(ht.sort, p.pred$fit) + matlines(ht.sort, p.CI) 

# Step-wise regression
step6 = lm(hip ~ cut(ht, 6))  # linear step wise reg fct with 6 cut points (i.e 7 intervals)
s.pred = predict(step6, newdata = list(ht = ht.sort), se = TRUE) 

s.CI = cbind(s.pred$fit - 2*s.pred$se.fit, s.pred$fit + 2*s.pred$se.fit)

plot(ht, hip) + lines(ht.sort, s.pred$fit) + matlines(ht.sort, s.CI)

# Splines
cuts = unname(quantile(ht))[2:4]  # using quartiles as knots

spline = lm(hip ~ bs(ht, degree = 1, knots = cuts))   # bs() fits a linear spline
sp.pred = predict(spline, newdata = list(ht = ht.sort), se = TRUE)
sp.CI = cbind(sp.pred$fit - 2*sp.pred$se.fit, sp.pred$fit + 2*sp.pred$se.fit)

plot(ht, hip) + lines(ht.sort, sp.pred$fit, col = 'red') + 
  matlines(ht.sort, sp.CI, col = 'red', lty = 3, lwd = 2)

# fit a cubic (df=3) smoothing spline, choosing lambda with cross-validation
# smoothing spline inherently uses all ht[i] as knots so no need to specify
smooth = smooth.spline(ht, hip, df = 3, cv = TRUE) 

plot(ht, hip) + lines(smooth)   # cubic smoothing spline looks very linear

# General Additive Models (GAMs)
# produces predictions using one of the following models:
gam = gam(hip ~ ns(Age, df = 5) +    # natural cubic spline     (Age as predictor)
                s(Thigh, df = 3) +   # smoothing spline         (Thigh as predictor) [note: s() is only used in gam()]
                ht,                  # simple linear regression (ht as predictor)
          data = seatpos)
par(mfrow = c(1,3))
plot(gam, se = TRUE)

# Logistic Regression
# data on whether or not an applicant is admitted based on gre, gpa, and rank of university
admit <- read.csv("https://www.maths.dur.ac.uk/users/hailiang.du/data/admit.csv")

pairs(admit, col = admit[,1] + 1)   # colour points based on admission status
admit$rank <- factor(admit$rank)    # change rank from num to factor
  
glm.fit <- glm(admit ~ ., data = admit, family = binomial)  # fits a logistic(/binomial) regression
gpred = predict(glm.fit, type = 'response') 

admit.pred = round(gpred)   # predicted value >=0.5 -> admittance  

errortable = table(admit.pred - admit$admit)
# 284 correct predictions, 116 missed predictions
# model has an error rate of 116/400 = 29% for these data, could be better

