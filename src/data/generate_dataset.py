import numpy as np
import pandas as pd
from datetime import datetime
import os
import random

def generate_dataset(n_customers=1000, n_loans=1200, output_dir='data/sample'):
    """
    Generate synthetic loan data with realistic NPL patterns.
    
    Parameters:
        n_customers: Number of unique customers
        n_loans: Total number of loans (some customers may have multiple)
        output_dir: Directory to save the generated CSV files
    """
    np.random.seed(42)  # For reproducibility
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate customer data
    customer_ids = [f'CUST{i:06d}' for i in range(1, n_customers + 1)]
    
    customers_df = pd.DataFrame({
        'customer_id': customer_ids,
        'age': np.random.normal(35, 10, n_customers).astype(int).clip(21, 65),
        'gender': np.random.choice(['M', 'F'], n_customers),
        'marital_status': np.random.choice(['Single', 'Married', 'Divorced', 'Widowed'], 
                                           n_customers, p=[0.3, 0.55, 0.1, 0.05]),
        'dependents': np.random.choice(range(5), n_customers, p=[0.2, 0.3, 0.3, 0.15, 0.05]),
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 
                                     n_customers, p=[0.4, 0.4, 0.15, 0.05]),
        'employment_type': np.random.choice(['Salaried', 'Self-Employed', 'Business Owner', 'Freelancer'], 
                                           n_customers, p=[0.6, 0.2, 0.1, 0.1]),
        'employment_years': np.random.exponential(5, n_customers).astype(int).clip(0, 30),
        'monthly_income': np.random.lognormal(15, 0.7, n_customers).astype(int),
        'existing_loans': np.random.choice(range(4), n_customers, p=[0.6, 0.3, 0.08, 0.02])
    })
    
    # Adjust monthly income based on education and employment
    income_factors = {
        'High School': 0.7,
        'Bachelor': 1.0,
        'Master': 1.5,
        'PhD': 2.0
    }
    
    for edu in income_factors:
        mask = customers_df['education'] == edu
        # Fix: Explicitly cast to integer after multiplication
        customers_df.loc[mask, 'monthly_income'] = (
            (customers_df.loc[mask, 'monthly_income'] * income_factors[edu]).astype(int)
        )
    
    # Generate loan application data
    loan_ids = [f'LOAN{i:06d}' for i in range(1, n_loans + 1)]
    
    # Prepare weights for customer selection (customers with more existing loans are more likely to get another)
    customer_weights = 1 + customers_df['existing_loans'].values
    customer_weights = customer_weights / customer_weights.sum()  # Normalize to sum to 1
    
    # Some customers may have multiple loans
    loan_customer_ids = np.random.choice(
        customer_ids, 
        size=n_loans, 
        replace=True, 
        p=customer_weights
    )
    
    # Define product types with their characteristics
    product_types = {
        'Auto': {'min_amount': 50000000, 'max_amount': 300000000, 'rate_mean': 0.1, 'rate_std': 0.02, 'term_range': (12, 60)},
        'Motorcycle': {'min_amount': 5000000, 'max_amount': 50000000, 'rate_mean': 0.15, 'rate_std': 0.03, 'term_range': (12, 36)},
        'Property': {'min_amount': 100000000, 'max_amount': 500000000, 'rate_mean': 0.08, 'rate_std': 0.01, 'term_range': (60, 180)}
    }
    
    # Base origination date (3 years ago from now)
    base_date = datetime.now() - timedelta(days=3*365)
    
    # Generate loan data
    loans_df = pd.DataFrame()
    loans_df['loan_id'] = loan_ids
    loans_df['customer_id'] = loan_customer_ids
    
    # Assign product types
    loans_df['product_type'] = np.random.choice(
        list(product_types.keys()), 
        size=n_loans, 
        p=[0.4, 0.4, 0.2]  # 40% Auto, 40% Motorcycle, 20% Property
    )
    
    # Generate loan details based on product type
    loan_amounts = []
    interest_rates = []
    tenor_months = []
    ltv_ratios = []
    
    for _, row in loans_df.iterrows():
        product = product_types[row['product_type']]
        
        # Get customer income from customers_df
        customer_income = customers_df.loc[
            customers_df['customer_id'] == row['customer_id'], 
            'monthly_income'
        ].values[0]
        
        # Loan amount based on product type and customer income
        max_affordable = customer_income * 36  # Max loan = 3 years of income
        min_loan = product['min_amount']
        max_loan = min(product['max_amount'], max_affordable)
        
        if max_loan < min_loan:
            max_loan = min_loan * 1.2  # Ensure we can give at least a small loan
            
        loan_amount = np.random.uniform(min_loan, max_loan)
        loan_amounts.append(int(loan_amount))
        
        # Interest rate based on product type
        rate = np.random.normal(product['rate_mean'], product['rate_std'])
        interest_rates.append(max(0.05, min(0.25, rate)))  # Clip to realistic range
        
        # Tenor based on product type
        min_term, max_term = product['term_range']
        tenor = np.random.choice(range(min_term, max_term + 1, 6))  # In 6-month increments
        tenor_months.append(tenor)
        
        # LTV ratio (higher for lower income customers)
        income_factor = 1 - (customer_income / customers_df['monthly_income'].max())
        ltv_base = 0.7 + (income_factor * 0.2)  # LTV between 70% and 90%
        ltv_ratios.append(min(0.9, max(0.5, ltv_base + np.random.normal(0, 0.05))))
    
    loans_df['loan_amount'] = loan_amounts
    loans_df['interest_rate'] = interest_rates
    loans_df['tenor_months'] = tenor_months
    loans_df['ltv_ratio'] = ltv_ratios
    
    # Calculate monthly installment (simplified)
    loans_df['monthly_installment'] = (
        loans_df['loan_amount'] * 
        (loans_df['interest_rate'] / 12) * 
        (1 + loans_df['interest_rate'] / 12) ** loans_df['tenor_months']
    ) / (
        (1 + loans_df['interest_rate'] / 12) ** loans_df['tenor_months'] - 1
    )
    
    # Origination dates (random dates in the past 3 years)
    origination_dates = [
        base_date + timedelta(days=np.random.randint(0, 365*3))
        for _ in range(n_loans)
    ]
    loans_df['origination_date'] = origination_dates
    
    # Branch and sales agent
    loans_df['branch_id'] = np.random.choice(range(1, 21), n_loans)  # 20 branches
    loans_df['sales_agent_id'] = np.random.choice(range(1, 101), n_loans)  # 100 sales agents
    
    # Generate payment history data (for the first 3 months)
    payments = []
    
    for _, loan in loans_df.iterrows():
        loan_id = loan['loan_id']
        orig_date = loan['origination_date']
        installment = loan['monthly_installment']
        
        # Customer risk profile (affects payment behavior)
        customer = customers_df[customers_df['customer_id'] == loan['customer_id']].iloc[0]
        
        # Calculate risk score (higher is riskier)
        risk_score = (
            (70 - customer['age']) * 0.02 +  # Younger customers are riskier
            (customer['dependents'] * 0.05) +  # More dependents = more financial pressure
            (3 - customer['employment_years']) * 0.1 +  # Less stable employment
            (customer['existing_loans'] * 0.15) +  # More existing debt
            (loan['ltv_ratio'] - 0.5) * 0.5 +  # Higher LTV is riskier
            np.random.normal(0, 0.2)  # Random factor
        )
        risk_score = max(0, min(1, risk_score))  # Normalize between 0 and 1
        
        # Generate payment records for first 3 months
        for mob in range(1, 4):
            payment_date = orig_date + timedelta(days=30*mob)
            
            # Payment behavior varies with risk score
            if mob == 1:
                # First payment - most customers pay on time
                # Fix: Ensure probabilities sum to 1
                prob_on_time = 0.9 - risk_score*0.4
                prob_slightly_late = risk_score*0.3
                prob_very_late = risk_score*0.1
                
                # Normalize probabilities to sum to 1
                total_prob = prob_on_time + prob_slightly_late + prob_very_late
                prob_on_time /= total_prob
                prob_slightly_late /= total_prob
                prob_very_late /= total_prob
                
                dpd_choice = np.random.choice(
                    [0, 1, 2],  # 0=on time, 1=slightly late, 2=very late
                    p=[prob_on_time, prob_slightly_late, prob_very_late]
                )
                
                if dpd_choice == 0:
                    dpd = 0
                elif dpd_choice == 1:
                    dpd = np.random.randint(1, 30)
                else:
                    dpd = np.random.randint(30, 60)
                
            elif mob == 2:
                # Second payment - behavior depends on first payment
                prev_dpd = [p['dpd'] for p in payments if p['loan_id'] == loan_id and p['mob'] == mob-1][0]
                if prev_dpd == 0:
                    # If first payment was on time
                    # Fix: Ensure probabilities sum to 1
                    prob_on_time = 0.95 - risk_score*0.3
                    prob_slightly_late = risk_score*0.25
                    prob_very_late = risk_score*0.05
                    
                    # Normalize
                    total_prob = prob_on_time + prob_slightly_late + prob_very_late
                    prob_on_time /= total_prob
                    prob_slightly_late /= total_prob
                    prob_very_late /= total_prob
                    
                    dpd_choice = np.random.choice(
                        [0, 1, 2],  # 0=on time, 1=slightly late, 2=very late
                        p=[prob_on_time, prob_slightly_late, prob_very_late]
                    )
                    
                    if dpd_choice == 0:
                        dpd = 0
                    elif dpd_choice == 1:
                        dpd = np.random.randint(1, 30)
                    else:
                        dpd = np.random.randint(30, 60)
                elif prev_dpd < 30:
                    # If first payment was late but under 30 days
                    # Fix: Ensure probabilities sum to 1
                    prob_improve = 0.4 - risk_score*0.2
                    prob_similar = 0.4
                    prob_worse = 0.2 + risk_score*0.2
                    
                    # Normalize
                    total_prob = prob_improve + prob_similar + prob_worse
                    prob_improve /= total_prob
                    prob_similar /= total_prob
                    prob_worse /= total_prob
                    
                    dpd_choice = np.random.choice(
                        [0, 1, 2],  # 0=improve, 1=similar, 2=worse
                        p=[prob_improve, prob_similar, prob_worse]
                    )
                    
                    if dpd_choice == 0:
                        dpd = 0
                    elif dpd_choice == 1:
                        dpd = prev_dpd + np.random.randint(-10, 20)
                    else:
                        dpd = prev_dpd + np.random.randint(20, 40)
                else:
                    # If first payment was very late
                    # Fix: Ensure probabilities sum to 1
                    prob_improve_a_lot = 0.2 - risk_score*0.15
                    prob_improve_a_bit = 0.3
                    prob_worsen = 0.5 + risk_score*0.15
                    
                    # Normalize
                    total_prob = prob_improve_a_lot + prob_improve_a_bit + prob_worsen
                    prob_improve_a_lot /= total_prob
                    prob_improve_a_bit /= total_prob
                    prob_worsen /= total_prob
                    
                    dpd_choice = np.random.choice(
                        [0, 1, 2],  # 0=improve a lot, 1=improve a bit, 2=worsen
                        p=[prob_improve_a_lot, prob_improve_a_bit, prob_worsen]
                    )
                    
                    if dpd_choice == 0:
                        dpd = 0
                    elif dpd_choice == 1:
                        dpd = prev_dpd - np.random.randint(5, 25)
                    else:
                        dpd = prev_dpd + np.random.randint(5, 30)
            else:  # mob == 3
                # Third payment - continued trajectory
                prev_payments = [p for p in payments if p['loan_id'] == loan_id]
                prev_dpd1 = prev_payments[0]['dpd']
                prev_dpd2 = prev_payments[1]['dpd']
                
                # Calculate trajectory
                trajectory = prev_dpd2 - prev_dpd1
                
                if trajectory <= -10:
                    # Improving payment behavior
                    dpd = max(0, prev_dpd2 - np.random.randint(10, 30))
                elif trajectory < 10:
                    # Stable payment behavior
                    dpd = max(0, prev_dpd2 + np.random.randint(-10, 10))
                else:
                    # Worsening payment behavior
                    dpd = prev_dpd2 + np.random.randint(0, 30)
                    
                    # Some loans may rapidly deteriorate
                    if risk_score > 0.7 and np.random.random() < 0.3:
                        dpd += np.random.randint(20, 40)
            
            # Calculate amount paid based on DPD
            if dpd == 0:
                amount_paid = installment
            elif dpd < 30:
                amount_paid = installment * np.random.uniform(0.7, 1.0)
            else:
                amount_paid = installment * np.random.uniform(0, 0.7)
                
            payment_status = 'Paid' if dpd == 0 else (
                'Partial' if amount_paid > 0 else 'Missed'
            )
            
            payments.append({
                'loan_id': loan_id,
                'payment_date': payment_date.strftime('%Y-%m-%d'),
                'mob': mob,
                'amount_due': installment,
                'amount_paid': amount_paid,
                'payment_status': payment_status,
                'dpd': max(0, dpd)  # Ensure DPD is non-negative
            })
    
    payments_df = pd.DataFrame(payments)
    
    # Generate loan outcomes for months 4-9
    outcomes = []
    
    for loan_id in loans_df['loan_id']:
        # Get loan's payment history for first 3 months
        loan_payments = payments_df[payments_df['loan_id'] == loan_id].sort_values('mob')
        
        # Skip if we don't have enough payment history
        if len(loan_payments) < 3:
            continue
            
        dpd1, dpd2, dpd3 = loan_payments['dpd'].values
        
        # Calculate trajectory
        trajectory = (dpd3 - dpd1) / 2
        
        # Determine future DPD values based on initial trajectory
        future_dpd = {}
        current_dpd = dpd3
        
        for mob in range(4, 10):
            # Add some randomness to trajectory
            if current_dpd < 30:
                # Good payers might stay good or occasionally slip
                adjusted_trajectory = trajectory * np.random.uniform(0.5, 1.5)
                if np.random.random() < 0.8:
                    # 80% chance good payers continue their trajectory
                    adjusted_trajectory = min(adjusted_trajectory, 10)
                    
            elif current_dpd >= 30 and current_dpd < 60:
                # Moderate risk payers could improve or worsen
                adjusted_trajectory = trajectory * np.random.uniform(0.8, 1.2)
                
                # Some will make efforts to improve
                if np.random.random() < 0.3:
                    adjusted_trajectory -= np.random.uniform(10, 30)
                
            else:  # current_dpd >= 60
                # High risk payers usually continue to deteriorate
                adjusted_trajectory = trajectory * np.random.uniform(0.9, 1.3)
                
                # Add chance of rapid deterioration
                if np.random.random() < 0.4:
                    adjusted_trajectory += np.random.uniform(5, 15)
                    
                # Add small chance of significant improvement (debt restructuring)
                if np.random.random() < 0.1:
                    adjusted_trajectory = -current_dpd * 0.8
            
            # Calculate next DPD value
            next_dpd = max(0, current_dpd + adjusted_trajectory)
            future_dpd[f'mob{mob}_dpd'] = int(next_dpd)
            current_dpd = next_dpd
        
        # Determine if loan ever reaches 90+ DPD in months 4-9
        max_dpd = max(future_dpd.values())
        ever_90plus = 1 if max_dpd >= 90 else 0
        
        outcomes.append({
            'loan_id': loan_id,
            **future_dpd,
            'max_dpd': max_dpd,
            'ever_90plus_dpd': ever_90plus
        })
    
    outcomes_df = pd.DataFrame(outcomes)
    
    # Save datasets to CSV
    customers_df.to_csv(f'{output_dir}/customers.csv', index=False)
    loans_df.to_csv(f'{output_dir}/loan_applications.csv', index=False)
    payments_df.to_csv(f'{output_dir}/payments.csv', index=False)
    outcomes_df.to_csv(f'{output_dir}/loan_outcomes.csv', index=False)
    
    # Print summary statistics
    print(f"Generated {len(customers_df)} customer records")
    print(f"Generated {len(loans_df)} loan applications")
    print(f"Generated {len(payments_df)} payment records")
    print(f"Generated {len(outcomes_df)} loan outcome records")
    print(f"NPL rate: {outcomes_df['ever_90plus_dpd'].mean():.2%}")
    
    # Return the dataframes for further analysis
    return customers_df, loans_df, payments_df, outcomes_df

if __name__ == "__main__":
    generate_dataset()