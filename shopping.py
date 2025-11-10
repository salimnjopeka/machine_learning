import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv('shopping_behavior_updated.csv')

# Remove unnecessary columns
df_clean = df.drop(columns=['Customer ID', 'Promo Code Used', 'Discount Applied', 'Subscription Status'])

print("="*80)
print("CONSUMER SHOPPING PREFERENCES ANALYSIS")
print("="*80)
print(f"Dataset: {df_clean.shape[0]} customers, {df_clean.shape[1]} features")
print()

# 1. AGE INFLUENCE ON SHOPPING PREFERENCES
print("1. AGE INFLUENCE ON SHOPPING PREFERENCES")
print("-" * 50)

# Create age groups for better analysis
df_clean['Age_Group'] = pd.cut(df_clean['Age'], 
                               bins=[0, 25, 35, 50, 65, 100], 
                               labels=['18-25', '26-35', '36-50', '51-65', '65+'])

# Age vs Purchase Amount
age_spending = df_clean.groupby('Age_Group')['Purchase Amount (USD)'].agg(['mean', 'median', 'std'])
print("Average Spending by Age Group:")
print(age_spending.round(2))
print()

# Age vs Product Categories
age_category = pd.crosstab(df_clean['Age_Group'], df_clean['Category'], normalize='index') * 100
print("Product Category Preferences by Age Group (%):")
print(age_category.round(1))
print()

# Age vs Frequency of Purchases
age_frequency = pd.crosstab(df_clean['Age_Group'], df_clean['Frequency of Purchases'], normalize='index') * 100
print("Shopping Frequency by Age Group (%):")
print(age_frequency.round(1))
print()

# Statistical test: Age vs Purchase Amount
young_spending = df_clean[df_clean['Age'] <= 35]['Purchase Amount (USD)']
old_spending = df_clean[df_clean['Age'] > 50]['Purchase Amount (USD)']
t_stat, p_value = stats.ttest_ind(young_spending, old_spending)
print(f"Statistical Test (Young vs Older): t-statistic = {t_stat:.3f}, p-value = {p_value:.3f}")
if p_value < 0.05:
    print("→ SIGNIFICANT difference in spending between age groups!")
else:
    print("→ No significant difference in spending between age groups")
print()

# 2. GENDER INFLUENCE ON SHOPPING PREFERENCES
print("2. GENDER INFLUENCE ON SHOPPING PREFERENCES")
print("-" * 50)

# Note: Dataset shows only Male customers - this limits gender analysis
gender_counts = df_clean['Gender'].value_counts()
print("Gender Distribution:")
print(gender_counts)
print("Note: Dataset contains only male customers - gender analysis limited")
print()

# Since all customers are male, we'll analyze other demographic patterns
# Let's look at item preferences by gender (even though it's all male)
gender_category = pd.crosstab(df_clean['Gender'], df_clean['Category'], normalize='index') * 100
print("Product Category Preferences by Gender (%):")
print(gender_category.round(1))
print()

# 3. LOCATION INFLUENCE ON SHOPPING PREFERENCES
print("3. LOCATION INFLUENCE ON SHOPPING PREFERENCES")
print("-" * 50)

# Top 10 states by customer count
top_states = df_clean['Location'].value_counts().head(10)
print("Top 10 States by Customer Count:")
print(top_states)
print()

# Average spending by top states
location_spending = df_clean.groupby('Location')['Purchase Amount (USD)'].agg(['mean', 'count']).round(2)
location_spending = location_spending[location_spending['count'] >= 5]  # States with at least 5 customers
location_spending_sorted = location_spending.sort_values('mean', ascending=False)
print("Average Spending by Location (States with 5+ customers):")
print(location_spending_sorted.head(10))
print()

# Category preferences by region (group states into regions)
state_to_region = {
    'California': 'West', 'Nevada': 'West', 'Oregon': 'West', 'Washington': 'West',
    'Arizona': 'West', 'Utah': 'West', 'Colorado': 'West', 'New Mexico': 'West',
    'Texas': 'South', 'Florida': 'South', 'Georgia': 'South', 'North Carolina': 'South',
    'South Carolina': 'South', 'Virginia': 'South', 'Alabama': 'South', 'Mississippi': 'South',
    'Louisiana': 'South', 'Arkansas': 'South', 'Tennessee': 'South', 'Kentucky': 'South',
    'West Virginia': 'South', 'Oklahoma': 'South',
    'New York': 'Northeast', 'Pennsylvania': 'Northeast', 'New Jersey': 'Northeast',
    'Massachusetts': 'Northeast', 'Connecticut': 'Northeast', 'Rhode Island': 'Northeast',
    'Vermont': 'Northeast', 'New Hampshire': 'Northeast', 'Maine': 'Northeast',
    'Illinois': 'Midwest', 'Ohio': 'Midwest', 'Michigan': 'Midwest', 'Indiana': 'Midwest',
    'Wisconsin': 'Midwest', 'Minnesota': 'Midwest', 'Iowa': 'Midwest', 'Missouri': 'Midwest',
    'North Dakota': 'Midwest', 'South Dakota': 'Midwest', 'Nebraska': 'Midwest', 'Kansas': 'Midwest',
    'Montana': 'West', 'Wyoming': 'West', 'Idaho': 'West', 'Alaska': 'West', 'Hawaii': 'West',
    'Delaware': 'Northeast', 'Maryland': 'Northeast'
}

df_clean['Region'] = df_clean['Location'].map(state_to_region)

# Regional analysis
region_category = pd.crosstab(df_clean['Region'], df_clean['Category'], normalize='index') * 100
print("Product Category Preferences by Region (%):")
print(region_category.round(1))
print()

region_spending = df_clean.groupby('Region')['Purchase Amount (USD)'].agg(['mean', 'median', 'count'])
print("Spending Patterns by Region:")
print(region_spending.round(2))
print()

# 4. SEASONAL PREFERENCES ANALYSIS
print("4. SEASONAL PREFERENCES ANALYSIS")
print("-" * 50)

season_category = pd.crosstab(df_clean['Season'], df_clean['Category'], normalize='index') * 100
print("Product Category Preferences by Season (%):")
print(season_category.round(1))
print()

season_spending = df_clean.groupby('Season')['Purchase Amount (USD)'].agg(['mean', 'median'])
print("Average Spending by Season:")
print(season_spending.round(2))
print()

# 5. ADVANCED INSIGHTS
print("5. ADVANCED INSIGHTS")
print("-" * 50)

# Age vs Season interaction
age_season = pd.crosstab(df_clean['Age_Group'], df_clean['Season'], normalize='index') * 100
print("Shopping Seasons by Age Group (%):")
print(age_season.round(1))
print()

# Payment method preferences by age
age_payment = pd.crosstab(df_clean['Age_Group'], df_clean['Payment Method'], normalize='index') * 100
print("Payment Method Preferences by Age Group (%):")
print(age_payment.round(1))
print()

# Shipping preferences by region
region_shipping = pd.crosstab(df_clean['Region'], df_clean['Shipping Type'], normalize='index') * 100
print("Shipping Preferences by Region (%):")
print(region_shipping.round(1))
print()

# 6. KEY FINDINGS SUMMARY
print("6. KEY FINDINGS SUMMARY")
print("-" * 50)

# Calculate some key metrics
young_avg = df_clean[df_clean['Age'] <= 35]['Purchase Amount (USD)'].mean()
old_avg = df_clean[df_clean['Age'] > 50]['Purchase Amount (USD)'].mean()
age_diff = abs(young_avg - old_avg)

# Most popular category overall
popular_category = df_clean['Category'].mode()[0]
popular_season = df_clean['Season'].mode()[0]

# Highest spending region
highest_region = region_spending['mean'].idxmax()
highest_region_avg = region_spending['mean'].max()

print(f"AGE INSIGHTS:")
print(f"• Average spending difference between young (≤35) and older (>50): ${age_diff:.2f}")
print(f"• Young customers (≤35) spend average: ${young_avg:.2f}")
print(f"• Older customers (>50) spend average: ${old_avg:.2f}")
print()

print(f"LOCATION INSIGHTS:")
print(f"• Highest spending region: {highest_region} (${highest_region_avg:.2f} average)")
print(f"• Most customers from: {top_states.index[0]} ({top_states.iloc[0]} customers)")
print()

print(f"GENERAL PATTERNS:")
print(f"• Most popular product category: {popular_category}")
print(f"• Most popular shopping season: {popular_season}")
print(f"• Dataset limitation: Only male customers (gender analysis not possible)")
print()

print("7. BUSINESS RECOMMENDATIONS")
print("-" * 50)
print("• Target different age groups with age-appropriate product categories")
print("• Customize marketing campaigns by region based on spending patterns")
print("• Adjust seasonal inventory based on category preferences by season")
print("• Consider payment method preferences when setting up checkout systems")
print("• Tailor shipping options based on regional preferences")
print("• Expand dataset to include female customers for complete gender analysis")

# Create visualizations
plt.figure(figsize=(20, 15))

# Plot 1: Age vs Purchase Amount
plt.subplot(3, 3, 1)
df_clean.boxplot(column='Purchase Amount (USD)', by='Age_Group', ax=plt.gca())
plt.title('Purchase Amount by Age Group')
plt.suptitle('')

# Plot 2: Category preferences by age
plt.subplot(3, 3, 2)
age_category.plot(kind='bar', stacked=True, ax=plt.gca())
plt.title('Category Preferences by Age Group')
plt.xticks(rotation=45)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Plot 3: Regional spending patterns
plt.subplot(3, 3, 3)
region_spending['mean'].plot(kind='bar', ax=plt.gca())
plt.title('Average Spending by Region')
plt.xticks(rotation=45)

# Plot 4: Seasonal category preferences
plt.subplot(3, 3, 4)
season_category.plot(kind='bar', stacked=True, ax=plt.gca())
plt.title('Category Preferences by Season')
plt.xticks(rotation=45)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Plot 5: Age distribution
plt.subplot(3, 3, 5)
df_clean['Age_Group'].value_counts().plot(kind='bar', ax=plt.gca())
plt.title('Customer Distribution by Age Group')
plt.xticks(rotation=45)

# Plot 6: Top states
plt.subplot(3, 3, 6)
top_states.head(8).plot(kind='bar', ax=plt.gca())
plt.title('Top 8 States by Customer Count')
plt.xticks(rotation=45)

# Plot 7: Payment methods by age
plt.subplot(3, 3, 7)
age_payment.plot(kind='bar', stacked=True, ax=plt.gca())
plt.title('Payment Method by Age Group')
plt.xticks(rotation=45)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Plot 8: Purchase amount distribution
plt.subplot(3, 3, 8)
plt.hist(df_clean['Purchase Amount (USD)'], bins=20, edgecolor='black', alpha=0.7)
plt.title('Purchase Amount Distribution')
plt.xlabel('Purchase Amount ($)')
plt.ylabel('Frequency')

# Plot 9: Review ratings by age group
plt.subplot(3, 3, 9)
df_clean.boxplot(column='Review Rating', by='Age_Group', ax=plt.gca())
plt.title('Review Rating by Age Group')
plt.suptitle('')

plt.tight_layout()
plt.show()

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("Charts show detailed patterns in consumer behavior across age, location, and seasons.")
print("="*80)