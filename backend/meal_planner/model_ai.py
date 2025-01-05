import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import KFold
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Embedding, Flatten, Concatenate, Input
from django.core.cache import cache
import json
from celery import shared_task
from typing import Dict, List, Any,Tuple
from multiprocessing import Pool
import random

class ImprovedContentBasedRecommender:
    def __init__(self):
        # Use a more sophisticated TF-IDF vectorizer
        self.tfidf = TfidfVectorizer(
            stop_words='english',
            min_df=2,  # Minimum document frequency
            max_df=0.95,  # Maximum document frequency
            ngram_range=(1, 2)  # Include bigrams
        )
        self.recipe_matrix = None
        self.recipes = None
        
    def fit(self, recipes_df):
        """Enhanced text processing with better type handling"""
        try:
            # Make a copy to avoid modifying the original
            recipes_df = recipes_df.copy()
            
            # Handle ingredients column - convert to list if string, or keep as list
            recipes_df['ingredients'] = recipes_df['ingredients'].apply(
                lambda x: x.strip('[]').split(', ') if isinstance(x, str) 
                else (x if isinstance(x, list) else [])
            )
            
            self.recipes = recipes_df
            
            # Create richer recipe texts with weighted features
            recipe_texts = recipes_df.apply(lambda x: (
                f"{x['name']} {x['name']} "  # Double weight for name
                f"{' '.join(str(ing) for ing in x['ingredients'])} {' '.join(str(ing) for ing in x['ingredients'])} "  # Double weight for ingredients
                f"{' '.join(str(c) for c in x['cuisine'])} "
                f"{' '.join(str(t) for t in x['tags'])} {' '.join(str(t) for t in x['tags'])} "  # Double weight for tags
                f"{' '.join(str(m) for m in x['meal_type'])} "
                f"{' '.join(str(d) for d in x['dish_type'])}"
            ), axis=1).values
            
            self.recipe_matrix = self.tfidf.fit_transform(recipe_texts)
            
        except Exception as e:
            print(f"Error in content recommender fit: {str(e)}")
            print("Recipes DataFrame columns:", recipes_df.columns)
            print("Ingredients column type:", recipes_df['ingredients'].dtype)
            print("Sample of ingredients:", recipes_df['ingredients'].head())
            raise

class ImprovedDeepLearningRecommender:
    def __init__(self, num_recipes, num_users):
        self.num_recipes = num_recipes
        self.num_users = num_users
        self.model = self._build_model()
        self.scaler_user = StandardScaler()
        self.scaler_recipe = StandardScaler()
        self.history = None

    def _build_model(self):
        # Simplified model architecture for better stability
        user_input = Input(shape=(11,), name='user_features')
        recipe_input = Input(shape=(8,), name='recipe_features')
        
        # User pathway - simplified
        user_dense = Dense(32, activation='relu')(user_input)
        user_dropout = Dropout(0.2)(user_dense)
        
        # Recipe pathway - simplified
        recipe_dense = Dense(32, activation='relu')(recipe_input)
        recipe_dropout = Dropout(0.2)(recipe_dense)
        
        # Combine pathways
        merged = Concatenate()([user_dropout, recipe_dropout])
        
        # Simplified deep layers
        x = Dense(64, activation='relu')(merged)
        x = Dropout(0.2)(x)
        x = Dense(32, activation='relu')(x)
        x = Dense(16, activation='relu')(x)
        
        # Output layer
        output = Dense(1, activation='sigmoid')(x)
        
        model = tf.keras.Model(inputs=[user_input, recipe_input], outputs=output)
        
        # Use a lower learning rate
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model

    def fit(self, user_features, recipe_features, ratings, validation_split=0.2):
        """
        Fixed fit method with separate scalers for user and recipe features
        """
        try:
            # Convert to numpy arrays if they're DataFrames
            if isinstance(user_features, pd.DataFrame):
                user_features = user_features.to_numpy()
            if isinstance(recipe_features, pd.DataFrame):
                recipe_features = recipe_features.to_numpy()
            
            # Check for NaN values
            if np.isnan(user_features).any() or np.isnan(recipe_features).any():
                print("Warning: Input features contain NaN values")
                # Fill NaN values with 0
                user_features = np.nan_to_num(user_features, 0)
                recipe_features = np.nan_to_num(recipe_features, 0)
            
            # Scale features using separate scalers
            user_features_scaled = self.scaler_user.fit_transform(user_features)
            recipe_features_scaled = self.scaler_recipe.fit_transform(recipe_features)
            
            # Convert ratings to binary (like/dislike) based on threshold
            binary_ratings = (ratings >= 3.5).astype(int)
            
            # Early stopping with patience
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )
            
            # Calculate class weights
            unique, counts = np.unique(binary_ratings, return_counts=True)
            total = len(binary_ratings)
            class_weights = {
                class_val: total / (len(unique) * count) 
                for class_val, count in zip(unique, counts)
            }
            
            # Train model
            self.history = self.model.fit(
                [user_features_scaled, recipe_features_scaled],
                binary_ratings,
                epochs=20,
                batch_size=32,
                validation_split=validation_split,
                callbacks=[early_stopping],
                class_weight=class_weights,
                verbose=1
            )
            
            return self.history
            
        except Exception as e:
            print(f"Error in fit method: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

class ImprovedHybridRecommender:
    def __init__(self, num_recipes, num_users):
        self.content_recommender = ImprovedContentBasedRecommender()
        self.dl_recommender = ImprovedDeepLearningRecommender(num_recipes, num_users)
        self.metrics = {}
        self.is_trained = False
    
    def _apply_rule_based_filtering(self, recipes_df, user_profile):
        """Apply rule-based filtering based on user preferences"""
        filtered_df = recipes_df.copy()
        
        # Initialize preference score
        filtered_df['preference_score'] = 0
        
        try:
            # Handle cuisine preferences
            if 'cuisine_preference' in user_profile:
                user_cuisines = user_profile['cuisine_preference']
                if isinstance(user_cuisines, str):
                    user_cuisines = eval(user_cuisines) if user_cuisines.startswith('[') else [user_cuisines]
                filtered_df['cuisine_match'] = filtered_df['cuisine'].apply(
                    lambda x: any(cuisine in str(x).lower() for cuisine in user_cuisines)
                )
                filtered_df.loc[filtered_df['cuisine_match'], 'preference_score'] += 1

            # Handle dietary preferences
            if 'dietary_preferences' in user_profile:
                dietary_prefs = user_profile['dietary_preferences']
                if isinstance(dietary_prefs, str):
                    dietary_prefs = eval(dietary_prefs) if dietary_prefs.startswith('[') else [dietary_prefs]
                
                # Map dietary preferences to recipe attributes
                pref_mapping = {
                    'VEG': 'vegetarian',
                    'VER': 'vegan',
                    'GLU': 'gluten_free',
                    'HAL': 'halal',
                    'PES': 'pescatarian'
                }
                
                for pref in dietary_prefs:
                    if pref in pref_mapping and pref_mapping[pref] in filtered_df.columns:
                        filtered_df.loc[filtered_df[pref_mapping[pref]] == True, 'preference_score'] += 1

            # Filter out disliked ingredients
            if 'disliked_ingredients' in user_profile and user_profile['disliked_ingredients']:
                disliked = user_profile['disliked_ingredients']
                if isinstance(disliked, str):
                    disliked = eval(disliked) if disliked.startswith('[') else [disliked]
                
                # Convert ingredients to lowercase for case-insensitive comparison
                disliked = [ing.lower() for ing in disliked]
                filtered_df['contains_disliked'] = filtered_df['ingredients'].apply(
                    lambda x: any(ing.lower() in str(x).lower() for ing in disliked)
                )
                filtered_df = filtered_df[~filtered_df['contains_disliked']]

            # Health goals based filtering
            if 'health_goals' in user_profile:
                goals = user_profile['health_goals']
                if isinstance(goals, str):
                    goals = eval(goals) if goals.startswith('[') else [goals]
                
                if 'LOS' in goals:  # Weight loss
                    filtered_df.loc[filtered_df['calories'] < user_profile['tdee'], 'preference_score'] += 1
                elif 'GAI' in goals:  # Weight gain
                    filtered_df.loc[filtered_df['calories'] > user_profile['tdee'] * 0.8, 'preference_score'] += 1

            return filtered_df
            
        except Exception as e:
            print(f"Error in rule-based filtering: {str(e)}")
            return recipes_df
    
    def _prepare_features(self, df):
        """
        Prepare features for the model with proper error handling
        """
        try:
            # Enhanced feature engineering
            user_features = pd.DataFrame({
                'age': df['age'],
                'bmi': df['bmi'],
                'tdee': df['tdee'],
                'is_male': (df['gender'] == 'male').astype(int),
                'goal_weight_loss': df['health_goals'].apply(lambda x: int('LOS' in str(x))),
                'goal_weight_gain': df['health_goals'].apply(lambda x: int('GAI' in str(x))),
                'goal_maintain': df['health_goals'].apply(lambda x: int('MAI' in str(x))),
                'veg_pref': df['dietary_preferences'].apply(lambda x: int('VEG' in str(x))),
                'vegan_pref': df['dietary_preferences'].apply(lambda x: int('VER' in str(x))),
                'gluten_free_pref': df['dietary_preferences'].apply(lambda x: int('GLU' in str(x))),
                'activity_level': df['activity_levels'].apply(
                    lambda x: {'SED': 0, 'LIG': 1, 'MOD': 2, 'VIG': 3, 'EXT': 4}.get(
                        str(x).strip('[]"\' ').split(',')[0].strip(), 2
                    )
                )
            })
            
            recipe_features = pd.DataFrame({
                'calories_ratio': df['calories'] / df['tdee'],
                'protein_ratio': df['protein'] / df['tdee'],
                'carbs_ratio': df['carbs'] / df['tdee'],
                'fat_ratio': df['fat'] / df['tdee'],
                'fiber': df['fiber'],
                'is_vegetarian': df['vegetarian'].astype(int),
                'is_vegan': df['vegan'].astype(int),
                'is_gluten_free': df['gluten_free'].astype(int)
            })
            
            return user_features, recipe_features
            
        except Exception as e:
            print(f"Error in _prepare_features: {str(e)}")
            print("\nDebugging information:")
            print(f"DataFrame columns: {df.columns}")
            print("\nSample of problematic columns:")
            for col in ['activity_levels', 'dietary_preferences', 'health_goals']:
                if col in df.columns:
                    print(f"\n{col} sample: {df[col].head()}")
            raise

    def train(self, recipes_df, merged_data, evaluate_after_training=False):
        """Train both recommenders with option to skip evaluation"""
        try:
            print("Training content-based recommender...")
            self.content_recommender.fit(recipes_df)
            
            print("Preparing features...")
            user_features, recipe_features = self._prepare_features(merged_data)
            ratings = merged_data['rating'].values
            
            print("Training deep learning recommender...")
            self.dl_recommender.fit(
                user_features,
                recipe_features,
                ratings,
                validation_split=0.2
            )
            
            self.is_trained = True
            print("Training complete!")
            
            if evaluate_after_training:
                print("Starting evaluation (this may take a while)...")
                return self.evaluate_async(merged_data, max_users=100)
            
        except Exception as e:
            print(f"Error in train method: {str(e)}")
            self.is_trained = False
            raise

    def _extract_first_value(self, array_string):
        """Extract first value from array-like string"""
        try:
            # Remove brackets and split
            cleaned = array_string.strip('[]').replace('"', '').replace("'", "").split(',')[0].strip()
            return cleaned
        except:
            return 'MOD'  # default value if parsing fails
    
    def create_meal_plans(self, recommendations, user_profile, n_recommendations=3):
        """
        Create meal plans from recommended recipes that match user's TDEE
        Returns a dictionary with meal types and their recommended recipes
        """
        try:
            meal_plans = {
                'breakfast': [],
                'lunch': [],
                'dinner': [],
                'snack': []
            }
            
            # Convert all meal types to lowercase for consistent comparison
            recommendations['meal_type'] = recommendations['meal_type'].apply(
                lambda x: [meal.lower() for meal in eval(str(x))] if isinstance(x, str) else [m.lower() for m in x]
            )
            
            # Calculate target calories per meal type
            daily_calories = user_profile['tdee']
            target_calories = {
                'breakfast': daily_calories * 0.25,  # 25% of daily calories
                'lunch': daily_calories * 0.35,      # 35% of daily calories
                'dinner': daily_calories * 0.30,     # 30% of daily calories
                'snack': daily_calories * 0.10       # 10% of daily calories
            }
            
            # Sort recommendations by final score in descending order
            sorted_recommendations = recommendations.sort_values('final_score', ascending=False)
            
            # Helper function to check if a recipe fits in the remaining calories
            def fits_calorie_budget(recipe_calories, meal_type, current_total):
                target = target_calories[meal_type]
                return (current_total + recipe_calories) <= (target * 1.1)  # Allow 10% flexibility
            
            # Process each meal type
            for meal_type in meal_plans.keys():
                current_calories = 0
                
                # Filter recipes for this meal type
                meal_recipes = sorted_recommendations[
                    sorted_recommendations['meal_type'].apply(lambda x: meal_type in x)
                ]
                
                # Get top 3 recipes that fit within calorie budget
                for _, recipe in meal_recipes.iterrows():
                    if len(meal_plans[meal_type]) >= 3:
                        break
                        
                    if fits_calorie_budget(recipe['calories'], meal_type, current_calories):
                        meal_plans[meal_type].append({
                            'recipe_id': recipe['recipe_id'],
                            'name': recipe['name'],
                            'calories': recipe['calories'],
                            'protein': recipe['protein'],
                            'carbs': recipe['carbs'],
                            'fat': recipe['fat'],
                            'final_score': recipe['final_score'],
                            'cuisine': recipe['cuisine']
                        })
                        current_calories += recipe['calories']
                    
                    # Ensure the total calories for all meal types do not exceed the user's TDEE
                    total_calories = sum(
                        sum(recipe['calories'] for recipe in meal_plans[meal_type])
                        for meal_type in meal_plans
                    )
                    if total_calories > daily_calories:
                        break
                
                # If we don't have enough recipes, fill with top recipes regardless of calories
                while len(meal_plans[meal_type]) < 3 and len(meal_recipes) > len(meal_plans[meal_type]):
                    recipe = meal_recipes.iloc[len(meal_plans[meal_type])]
                    meal_plans[meal_type].append({
                        'recipe_id': recipe['recipe_id'],
                        'name': recipe['name'],
                        'calories': recipe['calories'],
                        'protein': recipe['protein'],
                        'carbs': recipe['carbs'],
                        'fat': recipe['fat'],
                        'final_score': recipe['final_score'],
                        'cuisine': recipe['cuisine']
                    })
            
            # Ensure the total calories for any combination of one recipe from each meal type do not exceed the user's TDEE
            for i in range(n_recommendations):
                total_calories = sum(
                    meal_plans[meal_type][i]['calories'] if i < len(meal_plans[meal_type]) else 0
                    for meal_type in meal_plans if meal_type in ['breakfast', 'lunch', 'dinner', 'snack']
                )
                if total_calories > daily_calories:
                    for meal_type in meal_plans:
                        if i < len(meal_plans[meal_type]):
                            meal_plans[meal_type].pop(i)
            
            # Ensure each meal type has exactly 3 recipes
            for meal_type in meal_plans.keys():
                while len(meal_plans[meal_type]) < 3:
                    meal_plans[meal_type].append({
                        'recipe_id': None,
                        'name': 'Placeholder',
                        'calories': 0,
                        'protein': 0,
                        'carbs': 0,
                        'fat': 0,
                        'final_score': 0,
                        'cuisine': 'Placeholder'
                    })
            
            # Calculate and add nutritional summaries for each meal type
            for meal_type in list(meal_plans.keys()):  # Iterate over a copy of the keys
                total_calories = sum(recipe['calories'] for recipe in meal_plans[meal_type])
                total_protein = sum(recipe['protein'] for recipe in meal_plans[meal_type])
                total_carbs = sum(recipe['carbs'] for recipe in meal_plans[meal_type])
                total_fat = sum(recipe['fat'] for recipe in meal_plans[meal_type])
                
                meal_plans[f"{meal_type}_summary"] = {
                    'total_calories': total_calories,
                    'total_protein': total_protein,
                    'total_carbs': total_carbs,
                    'total_fat': total_fat,
                    'target_calories': target_calories[meal_type],
                    'calorie_difference': target_calories[meal_type] - total_calories
                }
            
            # Add daily totals
            daily_totals = {
                'total_calories': sum(summary['total_calories'] for meal_type, summary in meal_plans.items() if meal_type.endswith('_summary')),
                'total_protein': sum(summary['total_protein'] for meal_type, summary in meal_plans.items() if meal_type.endswith('_summary')),
                'total_carbs': sum(summary['total_carbs'] for meal_type, summary in meal_plans.items() if meal_type.endswith('_summary')),
                'total_fat': sum(summary['total_fat'] for meal_type, summary in meal_plans.items() if meal_type.endswith('_summary'))
            }
            meal_plans['daily_summary'] = {
                **daily_totals,
                'target_calories': daily_calories,
                'calorie_difference': daily_calories - daily_totals['total_calories']
            }
            
            return meal_plans
            
        except Exception as e:
            print(f"Error creating meal plans: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def get_recommendations(self, user_profile, n_recommendations=12):
        """Get personalized recipe recommendations for a user"""
        if not self.is_trained:
            print("Error: Model not trained. Please call train() first.")
            return pd.DataFrame()
            
        try:
            # Get all recipes
            all_recipes = self.content_recommender.recipes
            if all_recipes is None:
                print("Error: No recipes available. Make sure the model is trained.")
                return pd.DataFrame()

            # Apply rule-based filtering first
            filtered_recipes = self._apply_rule_based_filtering(all_recipes, user_profile)
            
            if filtered_recipes.empty:
                print("Warning: No recipes left after filtering. Returning original recommendations.")
                filtered_recipes = all_recipes

            # Prepare user features for deep learning
            user_features = pd.DataFrame({
                'age': [user_profile['age']],
                'bmi': [user_profile['bmi']],
                'tdee': [user_profile['tdee']],
                'is_male': [1 if user_profile['gender'] == 'male' else 0],
                'goal_weight_loss': [1 if 'LOS' in str(user_profile['health_goals']) else 0],
                'goal_weight_gain': [1 if 'GAI' in str(user_profile['health_goals']) else 0],
                'goal_maintain': [1 if 'MAI' in str(user_profile['health_goals']) else 0],
                'veg_pref': [1 if 'VEG' in str(user_profile['dietary_preferences']) else 0],
                'vegan_pref': [1 if 'VER' in str(user_profile['dietary_preferences']) else 0],
                'gluten_free_pref': [1 if 'GLU' in str(user_profile['dietary_preferences']) else 0],
                'activity_level': [2]  # Default to moderate activity
            })

            # Prepare recipe features
            recipe_features = pd.DataFrame({
                'calories_ratio': filtered_recipes['calories'] / user_profile['tdee'],
                'protein_ratio': filtered_recipes['protein'] / user_profile['tdee'],
                'carbs_ratio': filtered_recipes['carbs'] / user_profile['tdee'],
                'fat_ratio': filtered_recipes['fat'] / user_profile['tdee'],
                'fiber': filtered_recipes['fiber'],
                'is_vegetarian': filtered_recipes['vegetarian'].astype(int),
                'is_vegan': filtered_recipes['vegan'].astype(int),
                'is_gluten_free': filtered_recipes['gluten_free'].astype(int)
            })


            user_features = user_features.set_axis(range(user_features.shape[1]), axis=1) 
            recipe_features = recipe_features.set_axis(range(recipe_features.shape[1]), axis=1) 

            # Scale features
            user_features_scaled = self.dl_recommender.scaler_user.transform(user_features)
            recipe_features_scaled = self.dl_recommender.scaler_recipe.transform(recipe_features)

            # Get deep learning predictions
            predictions = self.dl_recommender.model.predict(
                [
                    np.repeat(user_features_scaled, len(recipe_features), axis=0),
                    recipe_features_scaled
                ],
                verbose=0
            )

            # Normalize scores to be between 0 and 1
            filtered_recipes['prediction_score'] = (predictions - predictions.min()) / (predictions.max() - predictions.min())
            
            if 'preference_score' in filtered_recipes.columns:
                # Normalize preference score
                filtered_recipes['preference_score'] = (filtered_recipes['preference_score'] - filtered_recipes['preference_score'].min()) / \
                    (filtered_recipes['preference_score'].max() - filtered_recipes['preference_score'].min())
                
                # Combine scores with weights that sum to 1
                filtered_recipes['final_score'] = (
                    filtered_recipes['prediction_score'] * 0.7 + 
                    filtered_recipes['preference_score'] * 0.3
                )
            else:
                filtered_recipes['final_score'] = filtered_recipes['prediction_score']

            # This ensures final_score will always be between 0 and 1
            recommendations = filtered_recipes.sort_values('final_score', ascending=False).head(n_recommendations)

            # Create and return meal plans
            meal_plans = self.create_meal_plans(recommendations, user_profile, n_recommendations) 
            return meal_plans

            

            # return recommendations[['recipe_id', 'name', 'cuisine', 'meal_type', 'calories', 
            #                      'protein', 'carbs', 'fat', 'final_score']]

        except Exception as e:
            print(f"Error getting recommendations: {str(e)}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()

    def precision_at_k(self,recommendations, relevant_items, k):
        """Calculates Precision@k."""
        recommendations = recommendations[:k]
        num_relevant_recommendations = len(set(recommendations) & set(relevant_items))
        return num_relevant_recommendations / k if k else 0

    def recall_at_k(self,recommendations, relevant_items, k):
        """Calculates Recall@k."""
        recommendations = recommendations[:k]
        num_relevant_recommendations = len(set(recommendations) & set(relevant_items))
        return num_relevant_recommendations / len(relevant_items) if relevant_items else 0

    def f1_score_at_k(self,recommendations, relevant_items, k):
        """Calculates F1-score@k."""
        precision = self.precision_at_k(recommendations, relevant_items, k)
        recall = self.recall_at_k(recommendations, relevant_items, k)
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0

    def dcg_at_k(self,recommendations, relevant_items, k):
        """Calculates Discounted Cumulative Gain (DCG)@k."""
        recommendations = recommendations[:k]
        dcg = 0
        for i, rec_id in enumerate(recommendations):
            if rec_id in relevant_items:
                dcg += 1 / np.log2(i + 2)  # +2 for relevance at rank 1
        return dcg

    def ndcg_at_k(self,recommendations, relevant_items, k):
        """Calculates Normalized Discounted Cumulative Gain (NDCG)@k."""
        dcg = self.dcg_at_k(recommendations, relevant_items, k)
        idcg = self.dcg_at_k(relevant_items, relevant_items, k)  # Ideal DCG
        return dcg / idcg if idcg else 0   

    def evaluate(self, merged_data, users_df, k=10):
        """Evaluates the recommender using the given metrics."""
        try:
            user_features, recipe_features = self._prepare_features(merged_data)
            ratings = merged_data['rating'].values
            all_metrics = []

            for user_id in merged_data['user_id'].unique():
                try:
                    user_data = merged_data[merged_data['user_id'] == user_id]
                    relevant_items = user_data[user_data['rating'] >= 3.5]['recipe_id'].tolist()
                    
                    if not relevant_items:
                        continue
                    
                    # Get user profile
                    user_profile = users_df[users_df['user_id'] == user_id].iloc[0].to_dict()
                    
                    # Get recommendations
                    meal_plans = self.get_recommendations(user_profile, n_recommendations=k)
                    
                    if not isinstance(meal_plans, dict):
                        continue

                    # Extract recipe IDs from meal plans
                    recommendations = []
                    for meal_type, recipes in meal_plans.items():
                        if meal_type != 'summary':  # Skip the summary key
                            if isinstance(recipes, list):
                                for recipe in recipes:
                                    if isinstance(recipe, dict) and 'recipe_id' in recipe:
                                        recommendations.append(recipe['recipe_id'])
                    
                    if not recommendations:
                        continue

                    # Calculate metrics
                    k_adj = min(k, len(recommendations))  # Adjust k if needed
                    precision = self.precision_at_k(recommendations, relevant_items, k_adj)
                    recall = self.recall_at_k(recommendations, relevant_items, k_adj)
                    f1 = self.f1_score_at_k(recommendations, relevant_items, k_adj)
                    ndcg = self.ndcg_at_k(recommendations, relevant_items, k_adj)

                    all_metrics.append({
                        'user_id': user_id,
                        'precision': precision,
                        'recall': recall,
                        'f1': f1,
                        'ndcg': ndcg
                    })
                    
                except Exception as user_exc:
                    continue

            # Calculate average metrics
            if all_metrics:
                avg_metrics = {
                    'precision': np.mean([m['precision'] for m in all_metrics]),
                    'recall': np.mean([m['recall'] for m in all_metrics]),
                    'f1': np.mean([m['f1'] for m in all_metrics]),
                    'ndcg': np.mean([m['ndcg'] for m in all_metrics]),
                }
                print("Final metrics:", avg_metrics)
                return avg_metrics
            else:
                print("No metrics could be calculated")
                return {'precision': 0, 'recall': 0, 'f1': 0, 'ndcg': 0}
                
        except Exception as e:
            print(f"Error in evaluate method: {str(e)}")
            print("Debug info:")
            print(f"merged_data shape: {merged_data.shape}")
            print(f"users_df shape: {users_df.shape}")
            raise

    def evaluate_async(self, merged_data, users_df, max_users=100):
        """Batch evaluation without multiprocessing"""
        try:
            # Sample users if needed
            unique_users = merged_data['user_id'].unique()
            if len(unique_users) > max_users:
                unique_users = random.sample(list(unique_users), max_users)

            # Process users in smaller batches
            batch_size = 10
            all_metrics = []
            
            for i in range(0, len(unique_users), batch_size):
                user_batch = unique_users[i:i + batch_size]
                batch_metrics = self._evaluate_batch(user_batch, merged_data, users_df)
                all_metrics.extend(batch_metrics)
                print(f"Processed {i + len(user_batch)}/{len(unique_users)} users")

            if all_metrics:
                avg_metrics = {
                    'precision': np.mean([m['precision'] for m in all_metrics]),
                    'recall': np.mean([m['recall'] for m in all_metrics]),
                    'f1': np.mean([m['f1'] for m in all_metrics]),
                    'ndcg': np.mean([m['ndcg'] for m in all_metrics]),
                }
                print("Final metrics:", avg_metrics)
                return avg_metrics
            
            return {'precision': 0, 'recall': 0, 'f1': 0, 'ndcg': 0}
            
        except Exception as e:
            print(f"Error in evaluate_async: {str(e)}")
            return {'precision': 0, 'recall': 0, 'f1': 0, 'ndcg': 0}

    def _evaluate_batch(self, user_ids, merged_data, users_df, k=10):
        """Evaluate a batch of users"""
        batch_metrics = []
        
        for user_id in user_ids:
            try:
                user_data = merged_data[merged_data['user_id'] == user_id]
                relevant_items = user_data[user_data['rating'] >= 3.5]['recipe_id'].tolist()
                
                if not relevant_items:
                    continue
                
                user_profile = users_df[users_df['user_id'] == user_id].iloc[0].to_dict()
                meal_plans = self.get_recommendations(user_profile, n_recommendations=k)
                
                if not isinstance(meal_plans, dict):
                    continue

                recommendations = []
                for meal_type, recipes in meal_plans.items():
                    if meal_type != 'summary' and isinstance(recipes, list):
                        recommendations.extend([r['recipe_id'] for r in recipes if isinstance(r, dict)])
                
                if recommendations:
                    k_adj = min(k, len(recommendations))
                    metrics = {
                        'user_id': user_id,
                        'precision': self.precision_at_k(recommendations, relevant_items, k_adj),
                        'recall': self.recall_at_k(recommendations, relevant_items, k_adj),
                        'f1': self.f1_score_at_k(recommendations, relevant_items, k_adj),
                        'ndcg': self.ndcg_at_k(recommendations, relevant_items, k_adj)
                    }
                    batch_metrics.append(metrics)
                    
            except Exception as e:
                print(f"Error evaluating user {user_id}: {str(e)}")
                continue
                
        return batch_metrics
