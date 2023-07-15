# Summary of Readings

## Options, Futures and Other Derivatives 

### Chapter 2: Futures Markets and Central Counterparties

Both future and forward contracts represent the obligation to buy an asset at a given price on a given date in the future. The key difference between the two is the futures contracts are exchange traded, whereas forward contracts are not. This means that the size of each futures contract is standardized (eg: 5000 bushels of corn) and the exchange mitigates credit risk by introducing or acting as a central counterparty. In addition to these differences, futures contracts are settled daily, in a mark-to-market fashion, whereas forward contracts are settled at the end of the contract. This means that after every trading day, for an underlying asset that has decreased in value, traders who are long futures contracts must cover their losses. This money will be transfered to traders who are short the contract. In practice, the majority of futures contracts are closed before their expiration so that the traders do not take delivery of the underlying asset. 

### Chapter 3: Hedging Strategieis Using Futures

#### Basic Principles

- Short Hedge: A short hedge is a hedge that involves a short position in a futures contract. It is appropriate when the hedger already owns an asset and expects to sell it some time in the future. For example, a farmer who already owns some hogs and knows he will sell them at the local market 2 months in the future could use a short hedge. An illustration of this: Consider an oil producer entering a contract to sell 1,000 barrels of oil on August 15th at the market price on that day. Also consider that the spot price today (on May 15th) is $50/barrel and the crude oil futures price for August delivery is $49/barrel. If the company enters a short position for 1,000 barrels for August delivery at $49/barrel today, this strategy should lock in $49/barrel. To see this, consider the scenarios where the spot price of crude increases and decreases: 
  - Spot price increases to $55/barrel: Then the company will lose ~$6/barrel on its short position, but gain ~$6/barrel on the long position. Resulting in a $49/barrel net
  - Spot price decreases to $45/barrel: Then the company will gain ~$4/barrel on the short position, but lose ~$4/barrel on the long position. Also resulting in a $49/barrel net
- Long Hedge: Hedges that involve taking long positions in a futures contract market. This is appropriate when a company knows it will have to purchase a certain asset in the future and wants to lock in a price now.