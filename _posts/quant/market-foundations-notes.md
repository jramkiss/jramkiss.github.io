# Fixed Income

## Treasury Debt Instruments

These are US debt instruments issued by the US Treasury that pay a semi-annual coupons at specified rate, followed by the final coupon payment and the principal payment at the maturity.

|         | Maturity            | Coupon   | Coupon Frequency | Principal          |
|---------|---------------------|----------|------------------|--------------------|
| T-bills | 4, 8, 13, 26, 52 weeks | None     | None             | Fixed              |
| T-notes | 2, 5, 7, 10 years      | Fixed    | Semiannual       | Fixed              |
| T-bonds | 20, 30 years            | Fixed    | Semiannual       | Fixed              |
| TIPS    | 5, 10, 30 years     | Fixed    | Semiannual       | Inflation adjusted |
| FRNs    | 2 years             | Floating | Quarterly        | Fixed              |
| STRIPS  | Derived             | None     | None             | Fixed              |

Reference: Mark Hendricks' notes and https://www.treasurydirect.gov/indiv/research/indepth/tbills/res_tbill.htm
 

The traditional debt instruments are T-bills, notes and bonds: 
- T-bills: $\le 1$ year maturity with no coupon (zero-coupon bonds)
- T-notes: 1, 5, 7, 10 year maturity with semi-annual coupon
- T-bonds: 20, 30 year maturity with semi-annual coupon

The US Treasury also issues Treasury Inflation Protected Securities (TIPS) for longer than 1 year maturity, which provide a hedge against inflation for investors. The coupon rate for a TIPS issue is fixed, but the face-value of the bond changes with CPI inflation. The fixed coupon rate is multiplied by an inflation-adjusted face-value, which leads to an inflation adjusted coupon. 

- Floating rate notes and STRIPS are also issued. FRN's have a floating coupon rate that provides a hedge against changing interest rates, while STRIPS are created by "stripping" the coupon and principal payments of an issue to create multiple zero-coupon bonds. This allows investors to trade zero-coupon bonds with maturities greater than 1 year. 

### Pricing Treasury Issues

Price is quoted per $100 face value. A bond trading above face-value is said to be trading above par. Bonds with a larger coupon rate than the current interest rate trade above par. As current interest rate increases, a bond with coupon rate, $c$, will become less attractive to investors, driving down the price. On the other hand, as interest rates decrease, this bond will become more attractive. 

The only factors impacting the price of a new issue is the time to maturity, $T$ and the coupon rate, $c$. In a frictionless market, the time to maturity would be irrelevant to the price. A fresh 5-year note would have the same price as a 10-year note issued 5 years ago with the same coupon. However, because of frictionality and liquidity issues, on-the-run bonds (i.e. freshly issued) tend to have higher liquidity and thus a higher price. 

### Discount Yield

The discount yield is important for quoting money-market prices but is not popular for pricing and research. **Treasury bills** are typically quoted as discount yields. For a price of $P$, face value of $\$100$ and no coupon, the discount yield is:

$$\text{discount yield} = \left(\frac{360}{n}\right)\frac{100-P}{100}$$

### Day-Count Conventions

The accrued interest is computed with a day-count convention

$$\text{accrued interest} = \frac{\text{days counted}}{\text{days in reference period}} \times \text{interest in reference period}$$

Common day-count conventions include:
* actual/actual: treasury notes and bonds
* 30/360: corporate and municipal bonds
* actual/360: money-market instruments (issued with 1yr or less, little credit risk)

Careful, or can lead to seeming arbitrage that doesn't exist.
* Hull's Business Snapshot 6.1 mentions case of T-bond vs Corp and former getting 1 day of accrual Feb 28 to Mar 1, while latter gets 3 days of accrual.


## Yied Curves and Discount Rates 

The yield-to-maturity (YTM) on a bond is merely a different way of quoting it's current market price. We can either talk about a bond's value as changes in price or changes in yields. It represents the average rate of return "received" if an investor were to purchase the bond at the quoted price and hold until maturity. For a bond, $j$, with maturity date, $T$, coupon rate, $c$, and coupon payments at times $t_i, 1 \le i \lt n$, we define the price function $P_j(t,T,c)$ as a function of the YTM, $y_j$:

$$
\begin{align*}
P_j(t,T,c) = \sum_{i=1}^{n-1}\frac{100\left(\frac{c}{2}\right)}{\left(1+\frac{y_j}{2}\right)^{2(T_i-t)}} + \frac{100\left(1+\frac{c}{2}\right)}{\left(1+\frac{y_j}{2}\right)^{2(T-t)}}
\end{align*}
$$

Where $t$ is the current/observed timepoint and $T-t$ is measured in years. Note that the same rate, $y_j$, is discounting cashflows at different maturities. **This makes the YTM (and price) unique to the coupon rate and term structure of security, $j$.**

**YTM has a non-linear and inverse relationship with bond price**. This can be seen from the formular above, but is easier to see on a zero-coupon bond: 

$$
\begin{align*}
P_j(t,T,0) = \frac{100}{\left(1+\frac{y_j}{2}\right)^{2(T-t)}}
\end{align*}
$$

<!-- ![](_imgs/price-vs-ytm.png) -->
<div class='figure' align="center">
    <img src="_imgs/price-vs-ytm.png" width="65%" height="65%">
    <div class='caption' width="70%" height="70%">
        <p> </p>
    </div>
</div>

For a coupon bond, YTM is not the return. This is becasue the forumla for YTM assumes we can re-invest the coupon payments at the YTM, which is not guaranteed. 


### The Spot Curve and No Arbitrage

Given a bond with scheduled coupon payments, we are able to calculated it's YTM for the current market price. However, as we have seen, this cannot be used to compare bonds with different coupon rates and payment structure. In addition, the YTM for a bond is an **average** discount rate applied to each cash flow. Consider 2 bonds with the same coupon rate, however one has 5 years to maturity and the other has 10. There is no difference in the first 4.5 years of cash flows received from these bonds, however because they have different maturities, their YTM will be different. Based on no-arbitrage pricing, the cost of these cash flows (cost can also be seen as discount rate) should be the same. We seek a rate which can be used across bond maturities to discount future cash flows. This is the spot rate. 

We seek a **spot curve** of discount rates, $r(t,T)$ such that we can price cashflows using:

$$
\begin{align*}
P_j(t,T,c) = \sum_{i=1}^{n-1}\frac{100\left(\frac{c}{2}\right)}{\left(1+\frac{r(t,T_i)}{2}\right)^{2(T_i-t)}} + \frac{100\left(1+\frac{c}{2}\right)}{\left(1+\frac{r(t,T)}{2}\right)^{2(T-t)}}
\end{align*}
$$

This differs from the YTM calculation since the spot rate, $r(t, T)$ does not depend on bond $j$ and is now a function of the cash flow timing, $T_i$. In the formula above, the spot rate is compounded semi-annually. We can compound this rate with any frequency, $f$ by converting from the current frequency, $c$ with:

$$
\begin{align}
r_{f} = f\left[\left(1+\frac{r_c}{c}\right)^{\frac{c}{f}}-1\right]
\end{align}
$$

For instance, changing the compounding from semiannually to daily would be,

$$
\begin{align}
r_{365} = 365\left[\left(1+\frac{r_2}{2}\right)^{\frac{2}{365}}-1\right]
\end{align}
$$

It is most common and convenient to use the continuously compounded rate, $r$, which can be written as a function of the $n$-times compounded rate as follows.

$$
\begin{align}
\displaystyle r &= n\ln\left(1+\frac{r_n}{n}\right)\\
\displaystyle r_n&= n\left(e^{\frac{r}{n}}-1\right)
\end{align}
$$

**Note:** We can only talk about an interest rate if we also mention the rate of compounding. The rate of compounding is like a distance metric. 

### The Discount Curve

The discount curve is a curve of discount factors across time. Discount factors, $Z(t,T)$, are linear factors that represent how much $1 is expected to be worth in the future. These are not compounded, but instead represent the discount for any compounded spot rate.

$$
\begin{align}
P_j(t,T) = Z(t,T) \times 100
\end{align}
$$


$$
\begin{align*}
\displaystyle P_j(t,T,c) = \sum_{i=1}^{n-1} 100 Z(t,T_i)\frac{c}{2} + 100 Z(t,T)\left(1+\frac{c}{2}\right)
\end{align*}
$$

If we estimate the discount curve (equivalent to estimating the spot curve) then we can apply these discount factors to price other fixed-income securities.

### Modelling the Spot Curve

Filter to eliminate...
* maturities that are too short or long
* quotes that do not have a quoted positive yield
* TIPS

Filter dates to eliminate...
* dates where no bond is maturing (identification)
* dates that are not benchmark treasury dates (liquidity)
  


To model the spot curve, we estimate the discount factors using the following notation
* $\boldsymbol{p}$: $n\times 1$ vector of the price for each issue
* $\boldsymbol{z}$: $k\times 1$ vector of the discounts for each cash-flow time
* $\boldsymbol{C}$: $n\times k$ matrix of the cashflow for each issue (row) and each time (column)

$$\boldsymbol{p} = \boldsymbol{C}\boldsymbol{z}$$

If we allow for estimation error and small market frictions:

$$\boldsymbol{p} = \boldsymbol{C}\boldsymbol{z}+\epsilon$$

We must first be careful to remove issues with maturities that are too long/short, quotes that do not have a quoted positive yield and TIPS. Dates where no bonds mature (identification) and dates that are not benchmark trasury date (liquidity) must also be filtered. Note that if a issue has a coupon falling on either of these dates, the entire issue must be removed. We cannot keep a fraction of the coupon dates for a given issue.

<div class='figure' align="center">
    <img src="_imgs/discount-curve.png" width="65%" height="65%">
    <div class='caption' width="70%" height="70%">
        <p> Estimated discount curve using </p>
    </div>
</div>


Recall that the **spot curve of interest rates** can be calculated for any **compounding frequency**. 

The figure below plots it for **continuous compounding**.

$$r(t,T) = -\frac{\ln\left(Z(t,T)\right)}{T-t}$$

#### Nelson-Seigel Model of the Discount Curve

# Nelson Siegel Model of the Yield Curve
_(taken verbatim from course notes)_

We need a model for the curve to avoid...
* missing data
* overfitting in-sample

The most famous such model is the **Nelson-Siegel** model. It models the spot curve according to 
* maturity
* 6 parameters


Let
* $\tau$ denote the maturity interval, $\tau=T-t$:
* $\boldsymbol{\theta}$ denote the vector of parameters to be estimated. (Here, 4 parameters: $\theta_0, \theta_1, \theta_2, \lambda$)

$$
\begin{align}
r(t,T) =& f(T-t,\boldsymbol{\theta})\\
=& \theta_0 + (\theta_1 + \theta_2)\frac{1-e^{-T/\lambda}}{T/\lambda} - \theta_2 e^{-T/\lambda}
\end{align}
$$

Note that for any set of parameters, we have...

| spot curve                  | $r_{\text{NS}}(t,T;\boldsymbol{\theta})$                                      |
|-----------------------------|-------------------------------------------------------------------------------|
| discount factors            | $Z_{\text{NS}}(t,T;\boldsymbol{\theta})$                                      |
| modeled bond prices         | $P_{\text{NS}}(t,T;\boldsymbol{\theta})$                                      |
| sum of squared model errors | $\left(P_{\text{mkt}}(t,T) - P_{\text{NS}}(t,T;\boldsymbol{\theta})\right)^2$ |


The Nelson-Siegel model is estimated by searching across the parameter space to minimize this sum of squared errors.

The **parameter table** below shows these optimized choices.

It shows this for Nelson-Siegel, as well as for an extended, 6-parameter, version of the model.

### Spot Curves VS YTM
_(taken verbatim from course notes)_

Note that we immediately calculate the YTM for each issue. If we plot these YTM against maturity, we get a curve. **This YTM curve is not the same as the spot curve, (often referred to as the yield curve, confusingly.)** The YTM is a certain average of semi-compounded spot rates over the range of the issue's maturity. For that reason, we should not be surprised to see the YTM plot is slightly below the spot curve for most of the range.

<div class='figure' align="center">
    <img src="_imgs/spot-curve-vs-ytm.png" width="65%" height="65%">
    <div class='caption' width="70%" height="70%">
        <p> Estimated spot curve and YTM </p>
    </div>
</div>

## Discount Curves, Rates and Factors

## Repo Agreements