# Market risk: Value at Risk and Expected Shortfall

## Definition of Value at Risk
Value at Risk (VaR) estimates the loss that a portfolio should not exceed over a given horizon with a given confidence level. A one-day 99% VaR of 10 million euros means that, under normal market conditions, the daily loss is expected to exceed 10 million euros on only about 1 day out of 100. VaR says nothing about the size of the loss on the days when the threshold is exceeded.

## Calculation methods
Historical simulation revalues the current portfolio under the market moves observed over a past window, for example the last 250 or 500 trading days, and reads the loss quantile from the resulting distribution. It makes no assumption on the shape of the distribution but depends heavily on the chosen window.
The parametric (variance-covariance) method assumes normally distributed returns and computes VaR from the volatilities and correlations of the risk factors. It is fast but underestimates fat tails and is poorly suited to options.
Monte Carlo simulation generates thousands of random scenarios from a stochastic model of the risk factors and fully revalues the portfolio in each scenario. It handles non-linear products but is the most computationally expensive method.

## Expected Shortfall
Expected Shortfall (ES), also called Conditional VaR, is the average loss in the scenarios beyond the VaR threshold. Unlike VaR, it captures the severity of tail losses and is a coherent risk measure, notably sub-additive. The Fundamental Review of the Trading Book (FRTB) replaced 99% VaR by a 97.5% Expected Shortfall for the calculation of market risk capital under the internal models approach.

## Backtesting
Backtesting compares the VaR forecast of each day with the actual profit and loss. A day on which the loss exceeds the VaR is called an exception. With a 99% VaR, about 2 to 3 exceptions are expected over 250 days; a significantly higher number indicates that the model underestimates risk and leads supervisors to increase the capital multiplier.
