import pandas as pd
import numpy as np
import random
import os

def generate_logical_dataset(num_samples=1000, output_path='dataset/student_burnout.csv'):
    print(f"Mulai generate {num_samples} data sintetik...")
    
    np.random.seed(42)
    random.seed(42)
    
    data = []
    
    for i in range(1, num_samples + 1):
        # Basic features
        grade = np.random.choice([9, 10, 11, 12], p=[0.25, 0.25, 0.25, 0.25])
        gender = np.random.choice(['Male', 'Female', 'Nonbinary'], p=[0.48, 0.48, 0.04])
        
        # We start with a base stress level
        base_stress = random.uniform(1, 3)
        
        # Sleep features
        sleep_hours = np.clip(np.random.normal(7, 1.5), 2.0, 12.0)
        
        # Screen time
        screen_time_hours = np.clip(np.random.normal(5, 2.0), 0.0, 16.0)
        
        # Academic load
        homework_hours = np.clip(np.random.normal(2.5, 1.5), 0.0, 12.0)
        tests_per_week = np.random.randint(0, 5)
        if grade == 12:  # Seniors have more tests
            tests_per_week += np.random.randint(1, 3)
            
        # Extracurricular
        num_activities = np.random.randint(0, 5)
        extracurricular_hours = num_activities * np.random.uniform(0.5, 1.5)
        
        # Support
        family_support = np.random.randint(1, 6)
        friend_support = np.random.randint(1, 6)
        teacher_support = np.random.randint(1, 6)
        
        # Commute
        commute_minutes = np.clip(np.random.normal(30, 20), 5.0, 180.0)
        
        # LOGIC INJECTION (Creating correlation)
        # 1. Stress modifier
        stress = base_stress
        if sleep_hours < 6: stress += 1.5
        if sleep_hours > 8: stress -= 1.0
        
        if homework_hours > 4: stress += 1.0
        if screen_time_hours > 6: stress += 0.5
        if tests_per_week > 3: stress += 1.0
        
        # Social support mitigates stress
        avg_support = (family_support + friend_support + teacher_support) / 3
        if avg_support >= 4: stress -= 1.0
        if avg_support <= 2: stress += 1.0
        
        # Calculate final self_rated_stress (1-5)
        self_rated_stress = int(np.clip(round(stress), 1, 5))
        
        # 2. Sleep Quality depends on screen time and stress
        sq = 5 - (stress * 0.5) - (screen_time_hours * 0.1)
        sleep_quality = int(np.clip(round(sq), 1, 5))
        
        # 3. Burnout Score (1-5)
        # Highly correlated with stress, low sleep quality, high workload
        burnout_calc = (
            (self_rated_stress * 0.4) + 
            ((6 - sleep_quality) * 0.3) + 
            ((homework_hours / 12) * 5 * 0.2) + 
            ((5 - avg_support) * 0.1)
        )
        
        # Add some noise
        burnout_calc += np.random.normal(0, 0.5)
        
        burnout_score = int(np.clip(round(burnout_calc), 1, 5))
        
        # High burnout flag for binary classification if needed
        high_burnout = 1 if burnout_score >= 4 else 0
        
        # Append row
        data.append({
            'student_id': i,
            'grade': grade,
            'gender': gender,
            'sleep_hours': round(sleep_hours, 1),
            'sleep_quality': sleep_quality,
            'homework_hours': round(homework_hours, 1),
            'tests_per_week': tests_per_week,
            'extracurricular_hours': round(extracurricular_hours, 1),
            'num_activities': num_activities,
            'screen_time_hours': round(screen_time_hours, 1),
            'commute_minutes': round(commute_minutes, 1),
            'family_support': family_support,
            'friend_support': friend_support,
            'teacher_support': teacher_support,
            'self_rated_stress': self_rated_stress,
            'burnout_score': burnout_score,
            'high_burnout': high_burnout
        })

    df = pd.DataFrame(data)
    
    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Dataset berhasil disimpan ke {output_path}!")
    
    # Simple check
    print("\nDistribusi Burnout Score:")
    print(df['burnout_score'].value_counts().sort_index())
    
    print("\nKorelasi dengan burnout_score (Top 5):")
    corr = df.select_dtypes(include=[np.number]).corr()['burnout_score'].sort_values(ascending=False)
    print(corr.head(6)[1:]) # skip self

if __name__ == '__main__':
    generate_logical_dataset()
