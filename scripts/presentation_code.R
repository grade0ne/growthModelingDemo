
divData <- read.csv("~/projectCSUN/csunThesisRepo/data/tempres/tempRes2.csv")

# Fig 1 - title crazy curves
fig1Data <- divData %>%
  mutate(across(c(Site, Leaf, Clone, Treatment), factor)) %>%
  group_by(Day)

ggplot(divData, aes(x = Day, y = Count, color = ID)) +
  geom_point(stat = "identity", position = position_jitterdodge(jitter.width = 4, dodge.width = 8)) +
  theme_classic() +
  theme(legend.position = "none")

# Fig 2 - linear growth

x <- seq(0, 10, length.out = 50)
y0 <- 5 # aka. "b", y-intercept
m <- 1

y <- (m * x) + y0 # y = mx + b

plot(x, y, type = "l", lwd = 2, main = "Linear Growth", col = "red")


# Fig 3 - exponential growth
x <- seq(0, 10, length.out = 50)
y0 <- 5
r <- 0.5

y <- y0 * exp(r * x)

plot(x, y, type = "l", lwd = 2, main = "Exponential Growth", col = "green")

# Fig 3a - intro to logistic

x <- seq(0, 10, length.out = 50)
y0 <- 5
alpha <- 0.002
r <- 1

y <- (r * y0) / (alpha * y0 + (r - alpha * y0) * exp(-r * x))
y_exp <- y0 * exp(r * x)

plot(x, y, type = "l", lwd = 2, main = "Logistic Growth", col = "blue", xlab = "Time", ylab = "N")
lines(x, y_exp, col = "red", lwd = 2, lty = 2)

# Fig 3b
x <- seq(0, 10, length.out = 50)
y0 <- 5
alpha <- 0.008
r <- 1

y <- (r * y0) / (alpha * y0 + (r - alpha * y0) * exp(-r * x))
y_exp <- y0 * exp(r * x)

plot(x, y, type = "l", lwd = 2, main = "Logistic Growth", 
     col = "blue", xlab = "Time", ylab = "N", ylim = c(0, 500))
lines(x, y_exp, col = "red", lwd = 2, lty = 2)
