
![consumer_behaviour.png](images/consumer_behaviour.png)
# Data-Driven-Insights-on-User-Behavior-and-Ad-Clicks

## BUSINESS OVERVIEW

In the digital age, advertising is a critical component of customer engagement and conversion. However, not all ads perform equally well, and understanding what drives user engagement is key to optimizing ad campaigns. By analyzing user demographics, behavior, and ad characteristics, businesses can tailor their strategies to maximize click-through rates (CTR) and improve return on investment (ROI).

Online advertising is a colossal business worth more than $50 billion, with advertisers increasingly focusing on targeted ad strategies. As the industry grows, it becomes essential to continuously measure the effectiveness of ads and refine strategies accordingly. One of the key metrics for measuring ad performance is the ad click, which indicates how often users interact with online advertisements. A higher number of ad clicks typically signifies a stronger interest in the advertised product, service, or content, making it a valuable indicator of user engagement and potential conversion.

## Problem Statement

Affiliate Marketing Company, Hubspot seeks to further invest significant resources in online advertising to drive user engagement, awareness, and conversions. However, the challenge often lies in determining which elements of an advertisement – such as content, targeting, timing, and format – are most effective in driving user interaction

## Objectives

**Main Objective:**

To optimize ad performance by understanding the factors that drive user engagement.

**Specific Objectives:**

1. Demographic Impact - Assess how demographic factors (e.g., age, gender, income) correlate with ad click-through rates.
2. Geographic Influence - Analyze the geographic distribution of ad clicks to identify regions with higher engagement.
3. Peak Click Times - Determine the optimal times of day and days of the week for ad delivery to maximize click-through rates.
4. User Behavior Patterns - Explore the relationship between user internet behavior and ad click propensity.
5. Ad Characteristics - Examine how the ad topic line impacts the likelihood of users clicking on the ad.

## Metrics of Success:
1. Click Through Rate (CTR)
   - The primary metric for measuring user engagement with ads. A good CTR (above 5.70%) indicates effective ad targeting and content. Monitoring CTR helps optimize ad strategies for higher engagement.
  
2. Statistical Significance
   - Through hypothesis testing, we will determine the impact of various demographic, behavioral, and ad-related factors on CTR. Identifying statistically significant factors allows Hubspot to focus on the most influential variables for improving ad performance.

## DATA UNDERSTANDING


The dataset used in this project is sourced from [this link](https://statso.io/wp-content/uploads/2023/01/ctr.zip). 

It contains 10,000 rows and 10 columns, capturing various features related to user interactions with online advertisements. The dataset includes a mix of numerical, categorical, and datetime variables, which are essential for analyzing patterns and factors influencing ad clicks.

Below is a description of each feature in the dataset:

| **Feature Name**            | **Description**                                                 | **Type**           | **Examples**               |
|-----------------------------|---------------------------------------------------------------|--------------------|----------------------------|
| Daily Time Spent on Site    | Time spent by a user on the site daily (in minutes)           | Continuous Numeric | 68.95, 80.23               |
| Age                         | Age of the user                                               | Continuous Numeric | 35, 23                     |
| Area Income                 | Average income of the user's geographical area (in dollars)  | Continuous Numeric | 55000, 72000               |
| Daily Internet Usage        | Daily usage of the internet by the user (in minutes)         | Continuous Numeric | 120.5, 96.8                |
| Ad Topic Line               | Topic headline of the ad viewed                              | Categorical Text   | "Top Ad Offer", "Great Deal" |
| City                        | City where the user resides                                   | Categorical Text   | "New York", "San Francisco" |
| Gender                      | Gender of the user                                            | Categorical        | Male, Female               |
| Country                     | Country of the user                                           | Categorical Text   | "Qatar", "India"             |
| Timestamp                   | Date and time of the ad interaction                          | DateTime           | 2024-10-31 14:53:00        |
| Clicked on Ad               | Whether the user clicked on the ad (1 = Yes, 0 = No)         | Binary Categorical | 1, 0                       |


## HYPOTHESIS TESTING
***


**1. Demographic Impact**
   
1.1 Age and Ad Clicks

- Null Hypothesis (H₀): There is no difference in ad click rates between different age groups.

- Alternative Hypothesis (H₁): There is a significant difference in ad click rates between different age groups.

1.2 Gender and Ad Clicks

- Null Hypothesis (H₀): There is no difference in the ad click rate between males and females.

- Alternative Hypothesis (H₁): There is a significant difference in the ad click rate between males and females.

1.3 Area Income and Ad Clicks

- Null Hypothesis (H₀): There is no relationship between area income (Area_Income) and the likelihood of clicking on an ad.

- Alternative Hypothesis (H₁): There is a significant relationship between area income and the likelihood of clicking on an ad.


**2. Geographic Influence**

2.1 Country and Ad Clicks

- Null Hypothesis (H₀): The country of the user does not affect the likelihood of clicking on an ad.

- Alternative Hypothesis (H₁): The country of the user significantly affects the likelihood of clicking on an ad.

2.2 Continent and Ad Clicks

- Null Hypothesis (H₀): The continent does not influence the likelihood of clicking on an ad.

- Alternative Hypothesis (H₁): The continent influences the likelihood of clicking on an ad.

**3. Peak Click Times**

3.1 Day of the Week and Ad Clicks

- Null Hypothesis (H₀): The ad click rate is the same across all days of the week.

- Alternative Hypothesis (H₁): The ad click rate differs across different days of the week.

3.2 Time of the Day and Ad Clicks

- Null Hypothesis (H₀): The ad click rate is the same across all hours of the day.

- Alternative Hypothesis (H₁): The ad click rate differs across different hours of the day.

**4. User Behavior**

4.1 Daily Internet Usage and Ad Clicks

- Null Hypothesis (H₀): There is no relationship between daily internet usage and the ad click rate.

- Alternative Hypothesis (H₁): There is a relationship between daily internet usage and the ad click rate.

4.2 Daily Time Spent on Site and Ad Clicks

- Null Hypothesis (H₀): There is no relationship between the daily time spent on the site and the ad click rate.

- Alternative Hypothesis (H₁): There is a relationship between the daily time spent on the site and the ad click rate.

**5. Ad Characteristics**

5.1 Ad Topic Line and Ad Click

- Null Hypothesis (H₀): The specific content of the ad topic line does not significantly affect the likelihood of clicking on the ad.
 
- Alternative Hypothesis (H₁): The specific content of the ad topic line significantly affects the likelihood of clicking on the ad.



***
## RESULTS AND RECOMMENDATIONS
***

**1. Target Ads Based on Age Demographics**

Since there is a positive relationship between age and ad clicks, you should tailor ads specifically for older age groups. For instance, creating ads that resonate more with older audiences or products that appeal to them might increase the likelihood of ad engagement.

**2. Gender-Specific Ad Targeting**

With women having a higher click rate, it's important to design ad campaigns that are more engaging for female users. This could involve personalizing the content, style, and messaging of ads to appeal specifically to women, potentially leading to higher engagement.

**3. Optimize Ad Timing for Maximum Engagement**

Given that certain days of the week (Wednesday, Saturday, and Monday) see higher ad engagement, you should consider scheduling your ad campaigns to target these days. Additionally, focusing on peak times like 9 AM, 12 PM, and 11 PM can further maximize ad effectiveness.

**4. Strategic Targeting for Different Countries and Continents**

The data suggests that country and continent have significant effects on ad clicks. Customize your ad campaigns based on geographic location, taking into account regional preferences and cultural differences. Ads should be localized to ensure relevance and increase engagement.

**5. Reevaluate Ad Content for Users with High Internet Usage**

The negative correlation between daily internet usage and ad clicks suggests that users who spend more time online may be less responsive to ads. For these users, you could consider showing less intrusive ads or offering rewards/incentives for engagement, such as discounts or exclusive content, to keep them interested and engaged with the ad content.

**6. A/B Testing**

Implement continuous A/B testing for ad creatives, targeting criteria, and campaign strategies to refine and optimize your approach based on the data insights.

