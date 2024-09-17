library(readxl)
data <- read_excel("DatasetProject.xlsx")
View(data)

library(tseries)
library(car)
library(nortest)
library(lmtest)
library(forecast)
library(tseries)
library(lmtest)
library(nortest)
library(TTR)
library(TSA)
library(car)

datats = ts(data[2], start=1975, end=2023)
plot(datats)

train_data <- window(datats, start=c(1975), end=c(2006))
test_data <- window(datats, start=c(2007))

# ARIMA
acf(train_data, lag.max = 20)
pacf(train_data, lag.max = 20)

d1 <- diff(train_data)
adf.test(d1)
acf(d1, lag.max = 20)
pacf(d1, lag.max = 20)

d2 <- diff(train_data, differences = 2)
adf.test(d2)
acf(d2, lag.max = 20)
pacf(d2, lag.max = 20)

model_1 <- arima(train_data, order = c(0, 1, 1))
coeftest(model_1)

model_2 <- arima(train_data, order = c(0, 2, 1))
coeftest(model_2)

predict_1 = predict(model_1, n.ahead=17)
predict_1

predict_2 = predict(model_2, n.ahead=17)
predict_2

forecast_values <- forecast(model_1, h = length(test_data))
accuracy(forecast_values, test_data)
err <- residuals(model_1)
Box.test(err, type = "Ljung-Box")
lillie.test(err)

forecast_values2 <- forecast(model_2, h = length(test_data))
accuracy(forecast_values2, test_data)
err2 <- residuals(model_2)
Box.test(err2, type = "Ljung-Box")
lillie.test(err2)

# Time Series Regression 
fit <- tslm(train_data ~ trend)

forecast_values <- forecast(fit, h = nrow(test_data))

train_fitted_values <- fit$fitted.values
rmse_train <- sqrt(mean((train_data - train_fitted_values)^2))
mape_train <- mean(abs((train_data - train_fitted_values) / train_data)) * 100

print(paste("Training RMSE: ", rmse_train))
print(paste("Training MAPE: ", mape_train))

rmse_test <- sqrt(mean((test_data - forecast_values$mean)^2))
mape_test <- mean(abs((test_data - forecast_values$mean) / test_data)) * 100

print(paste("Testing RMSE: ", rmse_test))
print(paste("Testing MAPE: ", mape_test))

# Neural Network
nn1 <- nnetar(train_data, p = 3, size = 5)

print(accuracy(nn1))

forecast_values <- forecast(nn1, h = length(test_data))

print(nn1$model)

test_accuracy <- accuracy(forecast_values, test_data)
print(test_accuracy)
