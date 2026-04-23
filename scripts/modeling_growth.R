# Created by Alex Mendelson
# for CSUN Biol551
# 23 April, 2026

#### I. Logistic Growth ----------------------------------------------- ####
# Try changing the y0, alpha, and r parameters of logistic growth below.
# The default parameters are y0=5, alpha=0.002, r=1
#   Blue line - actual population over time
#   Red line - exponential growth with the same r
#   Grey line - equilibrium population (r / alpha)
# 
# See if you can create the following scenarios:
# 
# 1. The population (N) grows exponentially, and never reaches an equilibrium
# 2. N starts high but drops to a lower equilibrium 
# 3. N doesn't change; population stays at equilibrium of N=300 
#     (hint: formula for EQ)

y0 <- 200
alpha <- 0.003
r <- 1

# Plot logistic
x <- seq(0, 10, length.out = 50)
y <- (r * y0) / (alpha * y0 + (r - alpha * y0) * exp(-r * x))
plot(x, y, type = "l", lwd = 2, main = "Logistic Growth", 
     col = "blue", xlab = "Time", ylab = "N", ylim = c(0, 500))

# Add exponential (r)
y_exp <- y0 * exp(r * x)
lines(x, y_exp, col = "red", lwd = 2, lty = 2)

# Add equilibrium
abline(h = r / alpha, lty = 5)


#### II. Experimental Data -------------------------------------------- ####

# 1. Paramecium (example) --------------------------------------------- ####
  # A single Paramecium culture was grown in a nutrient-limited 
  # microcosm and counted daily to track population growth as it 
  # approached carrying capacity.

# Assign your data object
paramecium_data <- read.csv("https://raw.githubusercontent.com/grade0ne/growthModelingDemo/main/paramecium.csv")

# Inspect its structure
library(tidyverse)
glimpse(paramecium_data)
plot(count~day, paramecium_data)

# The data look roughly logistic in shape, lets try fitting them to a logistic
# curve using growthrates:

install.packages("growthrates")
library(growthrates)

# First we'll use the built-in logistic function which estimates y0 (initial
# population size), mumax (maximum growth rate), and K (carrying capacity)

# Growthrates asks for a function (the equation), initial parameter guesses,
# and what your time and response variables are.

model1 <- fit_growthmodel(
  FUN = grow_logistic,
  p = c(y0 = 5, mumax = 0.5, K = 200),  # starting guesses, based on plotting the raw data
  time = paramecium_data$day,
  y = paramecium_data$count
)

summary(model1)
# In this summary, we can find the estimates of our three parameters: y0,
# mumax, and K. It also gives information about the uncertainty in these
# estimates (standard error, t, and p). Smaller standard errors give us more
# confidence in the fitted curve. The p-values here are about individual
# parameters, not about whether the whole model fits well, and they test
# whether each parameter is different from zero (so they are often not very
# informative in this context).

plot(model1)
# We can also plot the model (blue line) over the raw data to visually inspect
# how well it fits. This is often the easiest way to judge model fit, but you can
# also plot the residual spread: 

plot(residuals(model1) ~ obs(model1)[,2],
     xlab = "Observed count", ylab = "Residuals")
abline(h = 0, lty = 2)


# To estimate alpha, we have to write our own function to give to growthrates:
grow_logistic_alpha <- growthmodel(
  function(time, parms) {
    y0    <- parms[["y0"]]
    r     <- parms[["r"]]
    alpha <- parms[["alpha"]]
    
    y <- (r * y0) / (alpha * y0 + (r - alpha * y0) * exp(-r * time))
    
    cbind(time = time, y = y)
  },
  pnames = c("y0", "r", "alpha")
)

#...then use the same fit_growthmodels function as above to fit our paramecium
# data to this version of the logistic equation.

model2 <- fit_growthmodel(
  FUN = grow_logistic_alpha, # notice we specified our new equation here
  p = c(y0 = 5, r = 0.5, alpha = .002), # mumax and K replaced by r and alpha
  time = paramecium_data$day,
  y = paramecium_data$count
)

summary(model2)
plot(model2)
# When compared to the model1, r is very similar to mumax and y0 is largely
# unchanged, but we get alpha instead of K (much more biologically useful).
# We of course can still calculate the equilibrium (K) using r and alpha.


# 2. Duckweed (on your own) -------------------------------------------- ####
#   A small population of duckweed fronds was allowed to expand on a 
#   fixed-surface water container, with daily counts recording 
#   density-dependent growth over time.
#   
#   TASK: Find the intrinsic growth rate (r) and crowding effect (alpha) for
#   this population. Graph the model to visually assess model fit.

duckweed_data <- read.csv("https://raw.githubusercontent.com/grade0ne/growthModelingDemo/main/duckweed.csv")












