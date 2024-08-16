import numpy as np
from scipy.stats import norm

print("-----European Option Premium Calculator-----")                                  # Gets user input

Option = input("Are you purchasing a Call or a Put? : ")
S0 = float(input("Enter the inital price of a single share of the stock (ex: 100.0) : "))       # Initial stock price
K = float(input("Enter the desired Strike Price: "))                                            # Strike Price
Time = float(input("Enter the time until maturity (# of months): "))                            # Time to maturity
Rate = float(input("Enter the risk free interest rate (percent): "))                            # Risk free interest rate
sigma = float(input("Enter the Volatility of the stock (ex: 0.3): "))                            # Volatility of the stock

Sample = input("Is a sample size of 200,000 a reasonable amount? : ")                # Checks for adequate sample size
if Sample.lower() == "no":
    samplesize = float(input("Input a new sample size: "))
else:
    samplesize = 200000
                                                                              # Ask the user if the default confidence interval is acceptable
Confidence = input("The algorithm is utilizing a 95% Confidence Interval. Is this acceptable? : ")
if Confidence.lower() == "no":
    interval = float(input("Enter a new Confidence Interval (percent): "))
    myepsilon = (100 - interval) / 100
else:
    myepsilon = 0.05

# Data conversions
T = Time / 12
r = Rate / 100

#Pseudo Random Number Generator
rng = np.random.default_rng(12345)

# Function defined to calculate call option
def Call(S0, K, T, r, sigma, samplesize, myepsilon, rng):

    # Calculate call option price using the Black-Scholes formula 
    def black_scholes_call(S0, K, r, T, sigma):                             
        d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        tmp1 = S0 * norm.cdf(d1, loc=0, scale=1)
        tmp2 = K * np.exp(-r * T) * norm.cdf(d2, loc=0, scale=1)
        price = tmp1 - tmp2
        return price

    # Simulate terminal stock prices
    def terminal_stockprice(rng, S0, T, r, sigma, samplesize):
        mystandardnormalsample = rng.standard_normal(size=samplesize)
        tmp1 = (r - 0.5*(sigma ** 2)) * T                                   # It accounts for the average rate of return over time, adjusted for volatility.
        tmp2 = sigma * np.sqrt(T) * mystandardnormalsample                  # This term models the random fluctuations in stock prices, capturing the uncertainty.
        stockprice = S0 * np.exp(tmp1 + tmp2)
        return stockprice

    # Monte Carlo simulation for call option pricing
    def bs_call_mc(rng, S0, K, T, r, sigma, samplesize, myepsilon):
        # Generate terminal stock prices.
        mystockprice = terminal_stockprice(rng, S0, T, r, sigma, samplesize)  # Compute payoffs.
        payoffs = np.maximum(mystockprice - K, 0)  # Discount payoffs
        discountedpayoffs = np.exp(- r * T) * payoffs  # Compute MC price
        price = np.mean(discountedpayoffs)

        # Compute confidence interval next
        standarddev_rv = np.std(discountedpayoffs, ddof=1)
        standarddev_mcest = standarddev_rv / np.sqrt(samplesize)
        aepsilon = norm.ppf(1.0 - myepsilon * 0.5)

        # Left boundary of CI
        ci_left = price - aepsilon * standarddev_mcest

        # Right boundary of CI
        ci_right = price + aepsilon * standarddev_mcest
        return price, standarddev_mcest, ci_left, ci_right

    print('----------------')
    print('The analytical option price (Option Premium) is {:.4f}'.format(black_scholes_call(S0, K, r, T, sigma)))
    print('-----------------')
    MCresults = bs_call_mc(rng, S0, K, T, r, sigma, samplesize, myepsilon)
    print('MC price: {:.4f} and stdev of MC est: {:.4f}'.format(MCresults[0], MCresults[1]))
    print('CI based on MC is ({:.4f}, {:.4f})'.format(MCresults[2], MCresults[3]))

# Function defined to calculate call option
def Put(S0, K, T, r, sigma, samplesize, myepsilon, rng):

    def black_scholes_put(S0, K, r, T, sigma):
        d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        tmp1 = K * np.exp(-r * T) * norm.cdf(-d2, loc=0, scale=1)
        tmp2 = S0 * norm.cdf(-d1, loc=0, scale=1)
        price = tmp1 - tmp2
        return price

    def terminal_stockprice(rng, S0, T, r, sigma, samplesize):
        mystandardnormalsample = rng.standard_normal(size=samplesize)
        tmp1 = (r - 0.5 * (sigma ** 2)) * T
        tmp2 = sigma * np.sqrt(T) * mystandardnormalsample
        stockprice = S0 * np.exp(tmp1 + tmp2)
        return stockprice

    def bs_put_mc(rng, S0, K, T, r, sigma, samplesize, myepsilon):
        # Generate terminal stock prices.
        mystockprice = terminal_stockprice(rng, S0, T, r, sigma, samplesize)  # Compute payoffs.
        payoffs = np.maximum(K - mystockprice, 0)
        discountedpayoffs = np.exp(- r * T) * payoffs  # Compute MC price
        price = np.mean(discountedpayoffs)

        # Compute confidence interval next
        standarddev_rv = np.std(discountedpayoffs, ddof=1)
        standarddev_mcest = standarddev_rv / np.sqrt(samplesize)
        aepsilon = norm.ppf(1.0 - myepsilon * 0.5)

        # Left boundary of CI
        ci_left = price - aepsilon * standarddev_mcest

        # Right boundary of CI
        ci_right = price + aepsilon * standarddev_mcest
        return price, standarddev_mcest, ci_left, ci_right

    print('----------------')
    print('The analytical option price (Option Premium) is {:.4f}'.format(black_scholes_put(S0, K, r, T, sigma)))
    print('-----------------')
    MCresults = bs_put_mc(rng, S0, K, T, r, sigma, samplesize, myepsilon)
    print('MC price: {:.4f} and stdev of MC est: {:.4f}'.format(MCresults[0], MCresults[1]))
    print('CI based on MC is ({:.4f}, {:.4f})'.format(MCresults[2], MCresults[3]))


if Option.lower() == "call":
    Call(S0, K, T, r, sigma, samplesize, myepsilon, rng)

elif Option.lower() == "put":
    Put(S0, K, T, r, sigma, samplesize, myepsilon, rng)

