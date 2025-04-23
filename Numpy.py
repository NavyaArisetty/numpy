#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# ### Loading the dataset

# In[2]:


data = np.genfromtxt(r"C:\Users\17819\OneDrive\Desktop\numpy\employee_data.csv", delimiter=',',dtype=str,skip_header=1)


# In[3]:


with open(r"C:\Users\17819\OneDrive\Desktop\numpy\employee_data.csv") as f:
    header = f.readline().strip().split(',')


# In[4]:


print(header)


# In[5]:


print("First 100 rows of the dataset:")
print(data[:100])


# In[6]:


print("Dataset Preview:")
print(header)
for row in data[:100]:
    print(row)


# ### Cleaning the data

# In[7]:


employee_id = data[:, 0]
print(employee_id)


# In[8]:


hours = data[:, 1]
hours = np.where((hours == '') | (hours == 'N/A'), np.nan, hours)  
hours = hours.astype(float)  
hours = np.where((hours > 100) | (hours <= 0), np.nan, hours)
print(hours)


# In[9]:


tasks = data[:, 2]
tasks = np.where((tasks == 'None') | (tasks == ''), '0', tasks)  
tasks = tasks.astype(float)
print(tasks)


# In[10]:


salary = data[:, 3]
salary = np.where((salary == '') | (salary == 'N/A'), np.nan, salary) 
salary = salary.astype(float) 
salary = np.where(salary < 0, np.nan, salary)
print(salary)


# In[11]:


rating = data[:, 4]
rating = np.where((rating == 'None') | (rating == 'N/A') | (rating == ''), '0', rating)  
rating = rating.astype(int) 
print(rating)


# ### Handling the missing values

# In[12]:


mean_hours = np.nanmean(hours)
hours = np.where(np.isnan(hours), mean_hours, hours)
print(hours)


# In[13]:


tasks = np.where(np.isnan(tasks), 0, tasks)
print(tasks)


# In[14]:


mean_salary = np.nanmean(salary)
salary = np.where(np.isnan(salary), mean_salary, salary)
print(salary)


# In[15]:


header = ['Employee_ID', 'Hours_Worked', 'Tasks_Completed', 'Salary', 'Performance_Rating']


# In[16]:


data_cleaned = np.column_stack((employee_id, hours, tasks, salary, rating))


# In[17]:


final_data = np.vstack((header, data_cleaned))


# In[18]:


print("Cleaned Data (first 100 rows):")
print(final_data[:100])


# ### Data transforming

# In[19]:


salary_min = np.min(salary)
print(salary_min)


# In[20]:


salary_max = np.max(salary)
print(salary_max)


# In[21]:


common_salary = (salary - salary_min) / (salary_max - salary_min)
print("common (first 10 rows):")
print(common_salary[:10])


# In[22]:


rating_min = np.min(rating)
rating_max = np.max(rating)
common_rating = (rating - rating_min) / (rating_max - rating_min)
print("common Rating (first 10 rows):")
print(common_rating[:10])


# In[23]:


efficiency = np.divide(tasks, hours)
efficiency = np.where(hours == 0, np.nan, efficiency)
print("Efficiency (first 10 rows):")
print(efficiency[:10])


# In[24]:


cost_per_task = np.divide(salary, tasks)
cost_per_task = np.where(tasks == 0, np.nan, cost_per_task)
print("Cost per Task (first 10 rows):")
print(cost_per_task[:10])


# In[25]:


low_efficiency_mask = efficiency < 0.2
salary = np.where(low_efficiency_mask, np.nan, salary)
print("Salary after applying low efficiency mask (first 10 rows):")
print(salary[:10])


# ### Data filtering

# In[26]:


high_rating_mask = rating >= 4
high_rating_data = data[high_rating_mask]
print("Rows where Performance_Rating >= 4 (first 10 rows):")
print(high_rating_data[:10])


# In[27]:


no_outliers_mask = salary <= 100
filtered_data_no_outliers = data[no_outliers_mask]
print("Data without Salary > 100 (first 10 rows):")
print(filtered_data_no_outliers[:10])


# In[28]:


combined_condition_mask = (salary > 50) & (tasks > 10)
filtered_data_combined = data[combined_condition_mask]
print("Filtered data with salary > 50 and tasks > 10 (first 10 rows):")
print(filtered_data_combined[:10])


# In[29]:


salary = np.where(salary < 50, 50, salary)
print("Salary after replacing values < 50 with 50 (first 10 rows):")
print(salary[:10])


# ### Statistical analysis

# In[30]:


mean_salary = np.mean(salary)
mean_rating = np.mean(rating)
print("Mean Salary:", mean_salary)
print("Mean Rating:", mean_rating)


# In[31]:


median_salary = np.median(salary)
median_rating = np.median(rating)
print("Median Salary:", median_salary)
print("Median Rating:", median_rating)


# In[32]:


std_salary = np.std(salary)
std_rating = np.std(rating)
print("Standard Deviation of Salary:", std_salary)
print("Standard Deviation of Rating:", std_rating)


# In[33]:


min_salary = np.min(salary)
min_rating = np.min(rating)
print("Min Salary:", min_salary)
print("Min Rating:", min_rating)


# In[34]:


max_salary = np.max(salary)
max_rating = np.max(rating)
print("Max Salary:", max_salary)
print("Max Rating:", max_rating)


# In[35]:


correlation = np.corrcoef(salary, rating)
print("Correlation between Salary and Rating:")
print(correlation)


# In[36]:


rating_5_count = (rating == 5).sum()
print("Count of employees with rating 5:", rating_5_count)

