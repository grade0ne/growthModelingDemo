

#### I. Logistic Growth ----------------------------- ####
# Try changing the y0, alpha, and r parameters of logistic growth below.
# The default parameters are y0=5, alpha=0.002, r=1
#   Blue line - actual population over time
#   Red line - theoretical intrinsic growth (just r)
#   Grey line - equilibrium population
# 
# See if you can create the following scenarios:
# 
# 1. The population (N) grows exponentially, and never reaches an equilibrium
# 2. N starts high but drops to a lower equilibrium 
# 3. N doesn't change; population stays at equilibrium of N=300 
#     (hint: formula for EQ)

y0 <- 5
alpha <- 0.002
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


#### II. Experimental Data ----------------------------- ####
install.packages("growthrates")
library(growthrates)

# 1. Paramecium (example)
  # A single Paramecium culture was grown in a nutrient-limited 
  # microcosm and counted daily to track population growth as it 
  # approached carrying capacity.

