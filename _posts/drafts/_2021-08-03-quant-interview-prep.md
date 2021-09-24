---
layout: post
title: "Quant Interview Questions and Answers"
date: 2021-08-02 2:22
comments: true
author: "Jonathan Ramkissoon"
math: true
summary: 
---

Here is a list of practice questions I'm using to prepare for quant interviews. Any question with a single \* means that the concept of the answer in interesting and I should be familiar with it, any question with two \*'s means that I struggled a lot to understand the answer. The questions come from lots of different places, but mostly Heard on the Street by Timothy Falcon Crack and "A Practical Guide to Quant Finance Interviews". 


#### Problem 1 **

Given a properly shuffled deck of 52 cards, what is the expected number of draws until the first Ace if we pull without replacement? 


#### Solutions 1 

The key to this question is to start with a simpler version. 

- 5 cards, 4 are Aces:

$$ E(X) = \sum xP(x) = \frac{4}{5}(1) + \frac{1}{5}\frac{4}{4} (2) = \frac{6}{5} $$

- 6 cards, 4 are aces: 

$$ E(X) = \sum xP(x) = \frac{4}{6}(1) + \frac{2}{6}\frac{4}{5}(2) + \frac{2}{6}\frac{1}{5}\frac{4}{4}(3) = \frac{7}{5} $$

- 7 cards, 4 are Aces: 

$$ E(X) = \sum xP(x) = \frac{4}{7}(1) + \frac{3}{7}\frac{4}{6}(2) + \frac{3}{7}\frac{2}{6}\frac{4}{5}(3) + \frac{3}{7}\frac{2}{6}\frac{1}{5}\frac{4}{4}(4)= \frac{8}{5} $$

There's a pattern emerging here, where every time we add a non-Ace card, the expected value increases by $\frac{1}{5}$. So if we have 48 non-Ace cards, we can expect:

$$ E(X) = 1 + \frac{48}{5} = 10.6 $$


Alternative way: Consider a random non-Ace card, along with all the other 4 Aces. The probability of pulling this card before any other Ace is $\frac{1}{5}$. Since there are 48 other non-Ace cards in the deck, the expected number of cards pulled before we get an Ace is $48 \times \frac{1}{5} = 9.6$. Therefor the expected number of cards to pull in order to get an Ace is 10.6.

&nbsp;


#### Problem 2

What is the expected number of tosses to get three consecutive heads when flipping a fair coin?


#### Solution

X: number of tosses needed to get 3 H

What are the possible ways of getting 3 H?

- T + E(X): If we toss T, start over. This happens with probability $\frac{1}{2}$
- HT + E(X): Another option is to toss H then T, in which case we still have to start over. This happens with probability $\frac{1}{4}$
- HHT + E(X): Can also toss HHT, and we need to start over again. This happens with probability $\frac{1}{8}$
- HHH: It takes 3 tosses to get here. This happens with probability $\frac{1}{8}$

$$ E(X) = \frac{1}{2} (1 + E(X)) + \frac{1}{4}(2 + E(X)) + \frac{1}{8}(3 + E(X)) + \frac{1}{8}(3) $$

$$ E(X) = 14 $$

We can use this to come up with a general formular for the expected number of tosses to get $N$ head (or $N$ tails) in a row: 

$$ E(X) = \sum_{n=1}^{N} (\frac{1}{2})^2 (n + E(X)) + (\frac{1}{2})^NN $$

Which looks a lot more intimidating than it is. The sum term says that at each timestep we have a compounding $\frac{1}{2}$ probability of starting over, and if this happens then we have to add $n$, the number of tosses so far, to our count. We sum this for all possible $N$, because until we get all heads in a row, we always have a possibility of rolling tails and having to start over. 

&nbsp;


#### Problem 3

Suppose there are five pirates with 100 pieces of gold who are voting on splitting up the gold. The most senior pirate will propose a way to distribute the gold and requires 50\%50% of the votes in order to pass his proposal. If he receives less than 50\%50% of the votes then he will be killed and the process will begin with the next most senior pirate. How will the gold coins be divided?

#### Solution

&nbsp;


#### Problem 4 *

For a best of 3 set tennis match, would you bet on it finishing in two sets or three sets given that probabilty of winning a set is constant?

#### Solution

Let:

$$ P(A \text{ win}) = p $$
$$ P(B \text{ win}) = 1-p $$

Now there are 4 possible outcomes: AA, BB, AB, BA. AA and BB correspond to a 2-set match, BA and AB correspond to a 3-set match. For the 3-set matches, if we consider the outcome of the 3rd set only for the BA and AB matches, then the probabilities aren't balanced. 

Then:

$$P(2 \text{ set win}) = p^2 + (1 - p)^2 $$

$$P(3 \text{ set win}) = p(1-p) + p(1 - p) = 2p(1-p) $$

So, if $P(2 \text{ set win}) - P(3 \text{ set win}) > 0$, then we bet on the 3-set win.

$$P(2 \text{ set win}) - P(3 \text{ set win}) = p^2 + (1-p)^2 - 2p(1-p) $$

$$ 4p^2 - 4p + 1 = (1 - 2p)^2 $$

$(1 - 2p)^2 \ge 0$ so we always bet on the match finishing in 2 sets. 

&nbsp;

#### Problem 5 * 

You break a stick of length 1 at two points uniformly. What is the probability that the three resulting pieces can form a triangle?

#### Solution

This was a tricky problem for me. [This](https://www.youtube.com/watch?v=xFS4xpYQ82w) video explains a solution well.

We can't break the stick any way and get a triangle, so start off by figuring out the constraints for getting a triangle. What we need is for the sum of any 2 sides to be greater than the 3rd. If this isn't satisfied, we can't make the triangle. 

Imagining a line between 0 and 1, we'll call the first break $x$ and the 2nd break, $y$. 

So the length of each stick is: 
- $x$
- $y - x$
- $1 - y$

Now we have 2 options, $x > y$ or $y > x$, so we start with one of them, $x > y$:

Going back to the constraints:

$$ x + (y - x) > 1 - y $$

$$ x + 1 - y > y - x $$

$$ y - x + 1 - y > x $$

Simplifying these constraints gives us: 

- $ y > \frac{1}{2}$
- $x < \frac{1}{2}$
- $y < x +\frac{1}{2}$

Going back to the probability distributions, both $X$ and $Y$ are Uniformly distributed, so we can plot the joint density, which is a unit square. From here, we can plot the constraints and find the area of the region of interest, which will correspond to the probability we're interested in. 

For the case of $x > y$, the probability is $\frac{1}{8}$, and it's the same for the $y > x$. So the total probability is $\frac{1}{4}$.

&nbsp;

#### Problem 6

Consider a game where there are two blue balls and two red balls in a bag. Each ball is removed from the bag without replacement and if you guess the color of the ball correctly you earn a dollar. How much would you pay to play this game?

#### Solution

Draw a probability tree for this one. In the first roll, there's a $\frac{1}{2}$ chance of getting it right. The 2nd roll depends on the first - if we chose a Blue ball on the first roll, we have a $\frac{2}{3}$ of getting Red on the 2nd roll and vice versa. The 3rd roll is similar - if we chose BB in the first 2 rolls, we get the 3rd correct with probability 1. If we chose BR in the first 2 rolls, we get the 3rd correct with probability $\frac{1}{2}$. Then the 4th roll we always get correct. Here's the expected values for each choice: 

$$ E(X) = \frac{1}{2}(1) + \frac{1}{2}[\frac{2}{3} + \frac{2}{3}](1) + \frac{1}{2}[\frac{1}{3} + \frac{2}{3}(\frac{1}{2}) + \frac{2}{3}(\frac{1}{2}) + \frac{1}{3}](1) + 1(1) $$

Willing to pay no more than \$$\frac{17}{6}$.

&nbsp;

#### Problem 7

A calculating machine uses the digits 0 and 1 and transmits one of these digits through several stages. At every stage, there is a probability pp that the digit will be changed when it leaves and probability $q = 1 - p$, $q=1−p$ that it will not. Suppose the machine starts at state 0. What is the probability that the machine will display the correct digit after 2 stages?

#### Solution

&nbsp;

#### Problem 8

A transmitter is sending binary code (+ and - signals) that must pass through three relay signals before being sent on to the receiver. At each relay station, there is a 25% chance that the signal will be reversed. Suppose ++ signals make up 60% of the messages sent. If a ++ signal is received, what is the probability that a ++ was sent?

#### Solution

&nbsp;

#### Problem 9

There are N lions and 1 sheep in a field. All the lions really want to eat the sheep, but the problem is that if a lion eats a sheep, it becomes a sheep. A lion would rather stay a lion than be eaten by another lion. There is no other way for a lion to die than to become a sheep and then be eaten. When is it safe for any lion to eat?

#### Solution

&nbsp;

#### Problem 10

You roll a die up to three times. You can decide to stop and choose the number on the die (where the number is your payoff) during each roll. What’s your strategy?

#### Solution

&nbsp;

#### Problem 11 *

A hat contains 100 coins, 99 of them are guaranteed to be fair and 1 that has a $\frac{1}{2}$ chance to be double-headed. A coin is chosen at random and flipped 7 times. If it landed heads each time what is the probability that one of the coins is double-headed?

#### Solution

We can write some events: 

- $X$: One of the 100 coins is double headed
- $Y$: we flip 7 heads in a row

Then we're interested in $p(X \mid Y)$: 

$$ p(X \mid Y) = \frac{p(X, Y)}{p(Y)} $$

We know that: $p(X)$ = $\frac{1}{2}$

To calculate $p(X, Y)$, we need to find the possible ways to get 7 heads while we $X$ is true. There are 2 ways this can happen - we either choose the double headed coin with $\frac{1}{100}$ probability, or we choose another coin with $\frac{99}{100}$ probability and roll 7 heads.

$$ p(X, Y) = \frac{1}{2} \frac{1}{100} (1) + \frac{1}{2} \frac{99}{100} (\frac{1}{2})^7 $$

Now to calcualte $p(Y)$, we need to find all the possible ways of obtaining 7 heads in a row. This can happen one of 3 ways: 

- DH coin exists and we choose it
- DH coin exists and we don't choose it
- DH coin doesn't exist

$$ p(Y) = \frac{1}{2} \frac{1}{100} (1) + \frac{1}{2} \frac{99}{100} (\frac{1}{2})^7 + \frac{1}{2} (\frac{1}{2})^7 $$

So $p(X \mid Y) = \frac{p(X, Y)}{p(Y)} = 0.69$.


&nbsp;

#### Problem 12

A line of 100 passengers are boarding a plane. They each hold a ticket to one of the 100 seats on that flight. For convenience, the n-th passenger in line has a ticket for the seat number n. Being drunk, the first person in line picks a random seat. All of the other passengers are sober andw ill go to their proper seats unless it is already taken. In that case they will randomly choose a free seat. You're person number 100, what is the probability you end up in your seat?

#### Solution

&nbsp;

#### Problem 13

A casino offers a simple card game. There are 52 cards in a deck. Each time the cards are shuffled. You pick a card from the deck and the dealer picks another without replacement. If you have a larger number, you win; if the numbers are equal or yours is smaller, the house wins. What is your probability of winning?

#### Solution

&nbsp;

#### Problem 14

Two gamblers are playing a coin toss game. Gambler A has (n+1) coins, B has n fair coins. What is the probability that A will have more heads than B if both flip all their coins? 

#### Solution

&nbsp;

#### Problem 15

Four people, A, B, C and D need to get across a river. The only way is by an old bridge which holds at most 2 people at a time. Being dark, they can't cross the bridge without a torch, of which they only have 1. So each pair con only walk at the speed of the slower person. They need to get all of them across to the other side as quickly as possible. A is the slowest and takes 10 minutes to cross, B takes 5, C takes 2 and D takes 1. What is the minimum time to get all of them across to the other side? 

#### Solution

&nbsp;

#### Problem 16

A casino offers a game using a normal deck of 52 cards. The rule is that you turn over two cards each time. For each pair, if both are black, they go to the dealer's pile, if both are red they go to your pile. If one black and one red they are discarded. The process is repeated until you two go through all 52 cards. If you have more cards in your pile at the end, you win $100. How much are you willing to pay for this game?

#### Solution

&nbsp;

#### Problem 17

You have two burning ropes, each of which take 1 hour to burn. Both ropes have different densities so there's no guarantee of consistency in the time it takes different sections within the rope to burn. How do you use these ropes to measure 45 minutes?

#### Solution

&nbsp;

#### Problem 18 * 

How many trailing zeros are there in 100! ?

#### Solution

First question to ask is how to get a trailing zero? We have to multiply 2 and 5 in order to do this. So if we do a prime decomposition of 100! and count the number of 2's and 5's, that will be the number of times we multiply 2 and 5, which will be the number of trailing zeros.

There are 20 numbers in 100! that are divisible by 5 (100 / 5 = 20). We have to double count numbers that contain 5 twice, these are: 25, 50, 75 and 100. So in the prime decomposition of 100!, there are 24 occurrences of 5 and many more occurrences of 2. So there are 24 trailing zeros in 100!.

&nbsp;

#### Problem 19

There are 25 horses, each of which runs at a constant speed that is different from the others. Since the track only has 5 lanes, each race fcan have at most 5 horses. If you need to find the 3 fastest horses, what is the minimum number of races needed to identify them?

#### Solution

&nbsp;

#### Problem 20 **

If x^x^x^x... = 2, what is x?

#### Solution

&nbsp;

#### Problem 21

In unprofitable times corporations sometimes suspend dividend payments. Suppose that after a dividend has been paid the next one will be paid with probability 0.90.9, while after a dividend is suspended the next one will be suspended with probability 0.6. In the long run what is the fraction of dividends that will be paid?

#### Solution

&nbsp;

#### Problem 22

What is the expected number of rolls of a fair die needed to get all six numbers?

#### Solution

For the first number, we expect to roll once. The second number we expect to roll $\frac{6}{5}$ times because we have a $\frac{5}{6}$ probability of rolling a number we haven't seen before. Similar reasoning is used for the last 4 numbers: 

$$ E(X) = \frac{6}{6} + \frac{6}{5} + \frac{6}{4} + \frac{6}{3} + \frac{6}{2} + \frac{6}{1} $$

&nbsp;


#### Problem 23

A mother randomly selected two of her children to pick up a package from the post office. There was a 50% chance both children were daughters. How many daughters does this mother most likely have?

#### Solution

&nbsp;

#### Problem 24

You are in a city with streets on a perfect grid – every street is north-south or east-west. You are are driving north, and decide to randomly turn left or right with equal probability at the next 10 intersections.

After these 10 random turns, what is the probability you are still driving north?

#### Solution

&nbsp;

#### Problem 25

There are two trains that run between two cities. The trains are identical and run on identical routes, so passengers have no preference between the two and would take whichever train that pulls into the station. The trains run at the same frequency: exactly once an hour.

You often travel between the two cities on a whim, and when you do so, you show up at the station at a completely random time. Yet after many trips over the years, you notice that you have taken one of the trains three times as often as the other. Is this just really bad/good luck, or is there another likely explanation?

#### Solution

&nbsp;

#### Problem 26

You are driving to Seattle to meet some friends, and want to know whether you should bring an umbrella. You call up 3 of your friends who live there and independently ask them if it’s raining. Your friends like to mess with you, so each of them has a 1/3 chance of lying. If all 3 friends tell you it is raining, what is the probability it is actually raining there?

#### Solution

&nbsp;

#### Problem 27

Given $N$ points on a drawn randomly on the circumference of a circle, what's the probability that they all lie on a semi-circle?

#### Solution

&nbsp;

#### Problem 28 *

Given $X$ and $Y$ both come from $U \sim(0, 1)$. What is the probability that $XY < 0.5$?

#### Solution

This was a really interesting problem that I completely messed up in the beginning. 

We can consider the joint distribution, $f(x, y) = xy$. Since both $X$ and $Y$ are Uniform on (0, 1), we can plot the entire joint distribution support, which is a unit square. 

The area of this square that we're interested in is the area above $xy = 0.5$. Since $y = \frac{1}{2x}$ intersects the support at $x = \frac{1}{2}$, we have: 

$$ p(XY < \frac{1}{2}) = \frac{1}{2} - \int_\frac{1}{2}^1 \frac{1}{2x} dx $$

$$ = \frac{1}{2} [1 - ln(1) + ln(\frac{1}{2})] $$

$$ = \frac{1}{2} [1 - ln(\frac{1}{2})] $$

&nbsp;

#### Problem 29

You have many soccer teams, that will compete against each other in 5 rounds of elimination competitions until one team remains. You have 1000 dollars, how should you bet on each round of competition to ensure maximum profit? Found [here](https://www.wallstreetoasis.com/forums/crazy-de-shaw-phone-interview-question-help).

#### Solution

&nbsp;

#### Problem 30

There are n socks in a closet, and 3 of them are purple. What is the value of n such that when 2 socks are taken randomly from the drawer, the probability that they are both red is 0.5?

#### Solution

&nbsp;

#### Problem 31

(i) There are two traffic lights between your house and office. While going from your house to the office, you stop two times but while returning home from the office you stop only once. Given that the traffic lights are always red whenever you encounter them, how is this situation possible. He asked me to draw and explain my solution properly. (Hint: You don’t need to stop in a traffic light when you need to turn left)

#### Solution

&nbsp;

#### Problem 32

There is a 100-floor building and you have one egg. You need to find the lowest floor from which the egg breaks on dropping. 

#### Solution



&nbsp;

#### Problem 33

Given 10 stacks each stack contains 10 coins of 1 gram each. But one stack all coins with weight 9 gram. You have a weighing machine. You have to find the faulty stack in minimum number of weighings ? 

#### Solution

&nbsp;

#### Problem 34

Find Standard Deviation of a large array by dividing the array into multiple subarrays.

#### Solution

&nbsp;

#### Problem 35

Given an array of times(integer). Transfer that array to another side of the river in minimum time.(given that atmost two elements can travel on the boat at one time).

#### Solution

&nbsp;

#### Problem 36 (DE Shaw)

I have a response, $Y$ and 2 covariates, $x_1$ and $x_2$. When I regress $y$ on $x_1$, the $R^2$ is 0.01, and same for regressing on $x_2$. How does the $R^2$ change when I regress on $x_1$ and $x_2$?

#### Solution

Formula is [here](http://faculty.cas.usf.edu/mbrannick/regression/Part3/Reg2.html). But I need to learn how to derive these formulas and interpret the addition of a new variable. 

Prove that:

$$ R^2 = \beta_1 r_{yx_1} + \beta_2 r_{yx_2} = \frac{r_{yx_1}^2 - r_{yx_2}^2 - 2 r_{yx_1}r_{yx_2}r_{x_1x_2}}{1 - r_{x_1 x_2}^2} $$



&nbsp;

#### Problem 37 (DE Shaw)

I have 3 numbers, all uniformly distributed. I'll show you one and you decide whether or not to accept or reject. Only if you reject, I'll show you another and so on. The goal is to get the largest number. What is the strategy?


#### Solution

&nbsp;

#### Problem 38

$X_i, 1 \le i \le 4$ is log-Normally distributed with E(X) = 0 and Var(X) = 1. What is the probability that the product of all $X_i$ is less than 50?

#### Solution

&nbsp;

#### Problem 39

Two players, A and B, alternatively toss a fair coin (A tosses first and then B). The sequence of heads and tails is recorded. If there is a head followed by a tail (HT subsequence), the game ends and the person who tosses the tail wins. What is the probability that A wins the game?

#### Solution

Let:

- A: event that A wins after starting first
- B: event that B wins after starting first

We want to find $p(A)$.

$$ p(A) = \frac{1}{2}p(A \mid H) + \frac{1}{2}p(A \mid T) $$

Starting with $p(A \mid H)$, if A tosses a H, then the game starts over with B going first.

$$ p(A \mid H) = 1 - p(B) $$

By symmetry, p(A) = p(B). (Because of the way the events of A and B are set up). So now we have that:

$$ p(A \mid H) = 1 - p(A) $$

Now for $p(A \mid T)$. If A flips a T, then the next person to flip a H wins. So in order for A to win, B has to first flip a T with probability 0.5, then A has to be the first to flip a H from then on out. 

Let: 

- AH be the event that A flips a H first after starting
- BH be the event that B flips a H first after starting

$$ p(AH) = \frac{1}{2}(1) + \frac{1}{2}(1 - p(BH)) $$

With the same symmetry arguement used earlier, p(BH) = p(AH), so:

$$ p(AH) = \frac{1}{2}(1) + \frac{1}{2}(1 - p(AH)) $$

$$ p(AH) = \frac{2}{3} $$

Substituting $p(A \mid T)$ and $p(A \mid H)$ back into the first equation:

$$ p(A) = \frac{1}{2}(1 - p(A)) + \frac{1}{2}(\frac{1}{2}\frac{2}{3}) $$

$$ p(A) = \frac{4}{9} $$

&nbsp;

#### Problem 40 ***

Two people each tosses a fair coin N times. Find the probability that they get the same number of heads.

#### Solution

https://ryu577.github.io/jekyll/update/2018/10/08/competitive_coin_tossing.html

&nbsp;

#### Problem 41

9 fair coins and 1 special with double head. First I draw a coin and see the face is a head, what is the conditional probability that the next one is the special one?

#### Solution

&nbsp;

#### Problem 42

If the probability of seeing a shooting star over the course of an hour is 0.64, what is the probability of seeing a shooting star over the course of a half hour.

#### Solution

If we see at least 1 star in 1 hour then in 2 half hours:

- see 1 star in first half, none in 2nd half
- see none in 1st half, 1 in 2nd half
- see 1 star in both half hours

Let:

$$ p(\text{star in 30 minutes}) = x $$

$$ 2x(1-x) + x^2 = 0.64 $$

After applying the quadratic formula, $x = 0.4$.

&nbsp;

#### Problem 43 ***

I have three random variables X, Y , and Z with pairwise correlations all equal to r. Whar are the bounds on r? What are the bounds on corr(X, Z) if corr(X, Y ) = a and corr(Y, Z) = b?

#### Solution

- https://math.stackexchange.com/questions/284877/correlation-between-three-variables-question
- https://math.stackexchange.com/questions/1536234/correlation-coefficient-as-cosine
- http://jakewestfall.org/blog/index.php/2013/09/17/geometric-argument-for-constraints-on-corrxz-given-corrxy-and-corryz/
- https://stats.stackexchange.com/questions/72790/bound-for-the-correlation-of-three-random-variables
&nbsp;

#### Problem 44 ***

Let X and Y be two i.i.d uniform random variables drawn from (0,1). Let A be min(X,Y) and B be max(X,Y), what’s the correlation between A and B?

#### Solution

- https://math.stackexchange.com/questions/842264/finding-the-correlation-coefficient-of-ordered-statistics
- http://www.ams.sunysb.edu/~zhu/ams570/Lecture9_570.pdf


&nbsp;

#### Problem 45

Two people 𝐴 and 𝐵 throw fair coins independently. Let 𝑀 be the number of coin tosses until 𝐴 gets two consecutive heads. Let 𝑁 be the number of coin tosses until 𝐵 gets three consecutive heads. What is the probability that 𝑀>𝑁?

#### Solution

- https://math.stackexchange.com/questions/534800/probability-that-a-need-more-coin-tosses-to-get-two-consecutive-heads-than-b

&nbsp;

#### Problem 46

Let $X_1, \dots X_n$ be independent random variables uniformly distributed on [0,1]. Find $P(X_1 + \cdots + X_n \leq 1)$?

#### Solution

- https://math.stackexchange.com/questions/1683558/probability-that-sum-of-independent-uniform-variables-is-less-than-1


&nbsp;

#### Problem 47

a and b are randomly chosen real numbers in the interval [0, 1], that is both
a and b are standard uniform random variables. Find the probability that
the quadratic equation x2 + ax + b = 0 has real solutions.

#### Solution

Quadratic has real roots if the term in the square root is positive. Sketch joint distribution and integrate 


&nbsp;

#### Problem 48

How many times do we need to roll a fair die to get a sum divisible by 3?

#### Solution

Let X be the number of rolls until we get a sum divisible by 3. There are 2 outcomes after every roll:

1) sum mod 3 = 1
2) sum mod 3 = 2

In the first case, if we roll (2, 5) next, we get a sum that is divisible by 3. Similarly, in the second case, if we roll (1, 4) next, we get a sum divisible by 3. 

Let: 

- $X_1$ be the number of rolls until we get a 2 or 5
- $X_2$ be the number of rolls until we get a 1 or 4

$$ E(X) = \frac{1}{3}(1) + \frac{1}{3}(1 + E(X_1)) + \frac{1}{3}(1 + E(X_2)) $$

$$ E(X_1) = \frac{1}{3}(1) + \frac{2}{3}(1 + E(X_1)) $$
$$ E(X_1) = 3 $$
 
Therefore, $E(X_2) = 3$ and we can plug these back into the original equation to get $E(X) = 3$.

&nbsp;

#### Problem 49

What is the expected number of tosses of a die before we get 2 consecutive Heads or 2 consecutive Tails? What about 3 consecutive Heads or Tails?

#### Solution

&nbsp;

#### Problem 50 - Programming



#### Solution

&nbsp;

#### Problem 22

#### Solution

&nbsp;

#### Problem 22

#### Solution

&nbsp;

#### Problem 22

#### Solution

&nbsp;


## Concepts to brush up on

- OLS regression, all the bells and whistles here
- t-Tests for regression coefficients
- Formulas and interpretation of correlation and covariance
- Tests for Normality 
- 

