import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from django.core.cache import cache
import json
from celery import shared_task
from typing import Dict, List, Any,Tuple

class ContentBasedRecommender:
    def __init__(self):
        self.tfidf = TfidfVectorizer(stop_words='english', min_df=1, max_df=1.0)
        self.recipe_matrix = None
        self.recipes = None

    def fit(self, recipes_df):
        # Ensure all entries in 'ingredients' are strings
        recipes_df['ingredients'] = recipes_df['ingredients'].astype(str)
        
        # Strip brackets and split ingredients
        recipes_df['ingredients'] = recipes_df['ingredients'].str.strip('[]').str.split(', ')
        
        self.recipes = recipes_df
        recipe_texts = recipes_df.apply(lambda x: (
            f"{x['name']} {' '.join(x['ingredients'])} {' '.join(x['cuisine'])} "
            f"{' '.join(x['tags'])} {' '.join(x['meal_type'])} {' '.join(x['dish_type'])}"
        ), axis=1).values
        
        self.recipe_matrix = self.tfidf.fit_transform(recipe_texts)
        
    def get_recommendations(self, user_profile, n=50):
        user_text = (
            f"{' '.join(user_profile['dietary_preferences'])} {user_profile['health_goals']} {user_profile['cuisine_preference']} "
            f"{' '.join(user_profile['liked_ingredients'])}"
        )
        user_vector = self.tfidf.transform([user_text])
        sim_scores = cosine_similarity(user_vector, self.recipe_matrix)
        top_indices = sim_scores.argsort()[0][-n:][::-1]
        
        # Ensure all required columns are included in recommendations
        recommended_recipes = self.recipes.iloc[top_indices].copy()
        
        # Add missing dietary restriction columns if they don't exist
        dietary_columns = ['vegetarian', 'vegan', 'gluten_free', 'halal', 'pescatarian']
        for col in dietary_columns:
            if col not in recommended_recipes.columns:
                recommended_recipes[col] = False
        
        return recommended_recipes[['recipe_id', 'name', 'ingredients', 'cuisine', 'tags', 
                                  'meal_type', 'dish_type', 'calories', 'carbs', 'protein',
                                  'fat', 'fiber', 'gluten_free', 'halal', 'pescatarian',
                                  'vegetarian', 'vegan']]

class RuleBasedFilter:
    def apply_rules(self, user, recipes_df):
        filtered_df = recipes_df.copy()
        
        # Ensure all required dietary columns exist
        dietary_columns = {
            'vegetarian': False,
            'vegan': False,
            'gluten_free': False,
            'halal': False,
            'pescatarian': False
        }
        
        for col, default in dietary_columns.items():
            if col not in filtered_df.columns:
                filtered_df[col] = default

        # Prioritize dietary preferences but don't exclude
        preference_weights = {}  
        for index, row in filtered_df.iterrows():
            weight = 0
            dietary_prefs = user.get('dietary_preferences', [])
            
            # Map preference codes to column names
            pref_mapping = {
                'VEG': 'vegetarian',
                'VER': 'vegan',
                'GLU': 'gluten_free',
                'HAL': 'halal',
                'PES': 'pescatarian'
            }
            
            # Check dietary preferences
            for pref_code, column in pref_mapping.items():
                if pref_code in dietary_prefs and row.get(column, False):
                    weight += 1
            
            # Handle tag-based dietary preferences
            tag_preferences = [pref for pref in dietary_prefs 
                             if pref not in pref_mapping.keys()]
            if tag_preferences and any(pref in row['tags'] for pref in tag_preferences):
                weight += 1
                
            preference_weights[index] = weight

        filtered_df['preference_weight'] = pd.Series(preference_weights)
        filtered_df = filtered_df.sort_values('preference_weight', ascending=False)
        
        
        # weight on cuisines
        cuisine_weights = {}
        for index, row in filtered_df.iterrows():
            weight = 0
            cuisines = user.get('cuisine_preference', [])
            for cuisine in cuisines:
                if cuisine in (row['cuisine'] if isinstance(row['cuisine'], list) else []):
                    weight += 1
        
            cuisine_weights[index] = weight
        filtered_df['cuisine_weight'] = pd.Series(cuisine_weights)
        filtered_df = filtered_df.sort_values('cuisine_weight', ascending=False)
        

        # Filter disliked ingredients
        if user.get('disliked_ingredients'):
            disliked_ingredients = set(user['disliked_ingredients'])
            filtered_df = filtered_df[~filtered_df['ingredients'].apply(
                lambda x: bool(disliked_ingredients.intersection(set(x)))
            )]
            
        # If DataFrame is empty after filtering disliked ingredients, return original
        if filtered_df.empty:
            print("Warning: Filtering disliked ingredients resulted in an empty DataFrame. Returning original recommendations.")
            return recipes_df  

        return filtered_df

class DeepLearningRecommender:
    def __init__(self):
        self.model = self._build_model()
        self.scaler = StandardScaler()

    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            # tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse',metrics=['mae'])
        return model

    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y, epochs=20, batch_size=64, validation_split=0.2)

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

class HybridRecommender:
    def __init__(self):
        self.content_recommender = ContentBasedRecommender()
        self.rule_filter = RuleBasedFilter()
        self.dl_recommender = DeepLearningRecommender()
        self.metrics = {}
        self.relevance_threshold = 2.5
        self.feature_processors = None
        
        self.k_values = [5, 10, 20]  # Different k values for evaluation
        self.rating_threshold = 3.0  # Consider items with rating >= 4.0 as relevant
        self.minimum_interactions=5

    def fit(self, merged_data: pd.DataFrame) -> None:
        """
        Fit the hybrid recommender using merged data.
        
        Args:
            merged_data (pd.DataFrame): Combined data containing recipes, users, and ratings
        """
        print("Starting hybrid recommender fitting...")
        print("Initial merged data shape:", merged_data.shape)
        print("Available columns:", merged_data.columns.tolist())

        # Define and validate required columns
        required_columns = [
            'recipe_id', 'name', 'ingredients', 'cuisine', 'tags',
            'meal_type', 'dish_type', 'calories', 'fat', 'protein',
            'carbs', 'fiber', 'gluten_free', 'halal', 'pescatarian'
        ]

        # Validate and prepare data
        prepared_data = self._prepare_data(merged_data, required_columns)
        
        # Extract recipes data for content-based recommender
        recipes_data = self._prepare_recipes_data(prepared_data, required_columns)
        print("Fitting content-based recommender...")
        self.content_recommender.fit(recipes_data)

        # Prepare features for deep learning
        print("Preparing features for deep learning...")
        X = self._prepare_features(prepared_data)
        y = prepared_data['rating'].values

        # Train deep learning recommender
        print("Training deep learning recommender...")
        self.dl_recommender.fit(X, y)

        # Store training metrics
        history = self.dl_recommender.model.history.history
        self.metrics.update({
            'train_loss': history['loss'],
            'val_loss': history['val_loss'],
            'train_mae': history['mae'],
            'val_mae': history['val_mae']
        })
        
        print("Hybrid recommender fitting completed.")

    def _prepare_data(self, data: pd.DataFrame, required_columns: List[str]) -> pd.DataFrame:
        """Prepare and validate the input data."""
        prepared_data = data.copy()
        
        # Check for missing columns
        missing_columns = [col for col in required_columns if col not in prepared_data.columns]
        if missing_columns:
            print(f"Warning: Missing columns: {missing_columns}")
            self._add_missing_columns(prepared_data, missing_columns)

        # Convert numeric columns
        numeric_columns = ['calories', 'fat', 'protein', 'carbs', 'fiber', 
                         'weight', 'height', 'age', 'bmi', 'tdee', 'rating']
        for col in numeric_columns:
            if col in prepared_data.columns:
                prepared_data[col] = pd.to_numeric(prepared_data[col], errors='coerce').fillna(0)

        # Handle list columns
        list_columns = ['ingredients', 'cuisine', 'tags', 'meal_type', 'dish_type',
                       'dietary_preferences', 'activity_levels', 'health_goals']
        for col in list_columns:
            if col in prepared_data.columns:
                prepared_data[col] = prepared_data[col].apply(
                    lambda x: (x if isinstance(x, list) else
                             ([] if pd.isna(x) else
                              (x.split(',') if isinstance(x, str) else [])))
                )

        return prepared_data

    def _add_missing_columns(self, data: pd.DataFrame, missing_columns: List[str]) -> None:
        """Add missing columns with appropriate default values."""
        for col in missing_columns:
            if col in ['gluten_free', 'halal', 'pescatarian', 'vegan', 'vegetarian']:
                data[col] = False
            elif col in ['calories', 'fat', 'protein', 'carbs', 'fiber']:
                data[col] = 0.0
            elif col in ['ingredients', 'cuisine', 'tags', 'meal_type', 'dish_type']:
                data[col] = []
            else:
                data[col] = ''

    def _prepare_recipes_data(self, data: pd.DataFrame, required_columns: List[str]) -> pd.DataFrame:
        """Prepare recipe data for content-based recommendation."""
        recipes_data = data[required_columns].drop_duplicates(subset=['recipe_id'])
        return recipes_data

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for the deep learning model."""
        features = pd.DataFrame({
            'calories_ratio': df['calories'] / df['tdee'].replace(0, 1),  # Avoid division by zero
            'protein': df['protein'],
            'carbs': df['carbs'],
            'fat': df['fat'],
            'matches_preferences': df.apply(lambda x: int(any(
                pref in (x['tags'] if isinstance(x['tags'], list) else [])
                for pref in (x['dietary_preferences'] if isinstance(x['dietary_preferences'], list) else [])
            )), axis=1),
            'bmi': df['bmi'],
            'age': df['age'],
            'is_male': (df['gender'] == 'male').astype(int),
            'goal_weight_loss': df['health_goals'].apply(
                lambda x: int('LOS' in (x if isinstance(x, list) else []))
            ),
            'goal_weight_gain': df['health_goals'].apply(
                lambda x: int('GAI' in (x if isinstance(x, list) else []))
            ),
            'goal_maintain': df['health_goals'].apply(
                lambda x: int('MAI' in (x if isinstance(x, list) else []))
            )
        })
        
        # Handle NaN values
        features = features.fillna(0)
        return features

    def get_recommendations(self, user_profile: Dict) -> Dict:
        """Get personalized recipe recommendations for a user."""
        # Get initial recommendations from content-based filtering
        content_recs = self.content_recommender.get_recommendations(user_profile)
        
        # Apply rule-based filtering
        filtered_recs = self.rule_filter.apply_rules(user_profile, content_recs)
        
        # Prepare features for prediction
        features = self._prepare_features_for_prediction(user_profile, filtered_recs)
        
        # Get prediction scores
        scores = self.dl_recommender.predict(features)
        
        # Calculate recommendation metrics
        self.metrics['diversity_score'] = self._calculate_diversity(filtered_recs)
        self.metrics['novelty_score'] = self._calculate_novelty(filtered_recs, user_profile)
        
        # Generate meal plans
        return self._create_meal_plans(filtered_recs, scores,user_profile['tdee'])

    def _prepare_features_for_prediction(self, user_profile: Dict, recipes_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for prediction on new data."""
        features = pd.DataFrame({
            'calories_ratio': recipes_df['calories'] / user_profile.get('tdee', 2000),  # Default TDEE
            'protein': recipes_df['protein'],
            'carbs': recipes_df['carbs'],
            'fat': recipes_df['fat'],
            'matches_preferences': recipes_df['tags'].apply(
                lambda x: int(any(pref in (x if isinstance(x, list) else [])
                               for pref in user_profile.get('dietary_preferences', [])))
            ),
            'bmi': user_profile.get('bmi', 0),
            'age': user_profile.get('age', 0),
            'is_male': int(user_profile.get('gender', '') == 'male'),
            'goal_weight_loss': int('LOS' in user_profile.get('health_goals', [])),
            'goal_weight_gain': int('GAI' in user_profile.get('health_goals', [])),
            'goal_maintain': int('MAI' in user_profile.get('health_goals', []))
        })
        
        # Handle NaN values
        features = features.fillna(0)
        return features

    def _create_meal_plans(self, recipes_df: pd.DataFrame, scores: np.ndarray, user_tdee: float) -> Dict[str, List[str]]:
        """Create meal plans from recommended recipes that match the user's TDEE."""
        
        # Copy recipes and add the score
        recipes_df = recipes_df.copy()
        recipes_df['score'] = scores
        recipes_df = recipes_df.sort_values('score', ascending=False)

        meal_plans = {
            'breakfast': [],
            'lunch': [],
            'dinner': [],
            'snack': []
        }

        # Assume rough calorie distribution across meals
        meal_distribution = {
            'breakfast': 0.25,  # 25% of TDEE for breakfast
            'lunch': 0.35,      # 35% of TDEE for lunch
            'dinner': 0.3,      # 30% of TDEE for dinner
            'snack': 0.1        # 10% of TDEE for snack
        }

        available_recipes = recipes_df.copy()

        for meal_type in meal_plans.keys():
            meal_recipes = available_recipes[
                available_recipes['meal_type'].apply(lambda x: meal_type in (x if isinstance(x, list) else []))
            ]
            
            # Calories target for this meal type
            meal_target_calories = user_tdee * meal_distribution[meal_type]

            # Select recipes that get close to the calorie target
            selected_recipes = []
            total_calories = 0
            
            for _, recipe in meal_recipes.iterrows():
                recipe_calories = recipe['calories']  # Assuming 'calories' column in DataFrame
                
                # Check if adding this recipe would exceed the target
                if total_calories + recipe_calories <= meal_target_calories:
                    selected_recipes.append(recipe['name'])
                    total_calories += recipe_calories
                    
                    # Remove selected recipe from the pool
                    available_recipes = available_recipes[available_recipes['name'] != recipe['name']]

                # Stop if total calories get close enough to the target
                if total_calories >= meal_target_calories * 0.95:  # Allow for a 5% tolerance
                    break
            
            # Store selected recipes for the meal type
            meal_plans[meal_type] = selected_recipes

        return meal_plans


    
    def _calculate_diversity(self, recommendations: pd.DataFrame) -> float:
        # Implement diversity calculation
        unique_cuisines = len(set(recommendations['cuisine'].sum()))
        unique_ingredients = len(set(recommendations['ingredients'].sum()))
        return (unique_cuisines + unique_ingredients) / (2 * len(recommendations))

    def _calculate_novelty(self, recommendations:pd.DataFrame, user_profile: Dict) -> float:
        # Implement novelty calculation
        known_ingredients = set(user_profile['liked_ingredients'])
        new_ingredients = set(recommendations['ingredients'].sum()) - known_ingredients
        return len(new_ingredients) / len(recommendations['ingredients'].sum())
    
    def evaluate(self, test_data:pd.DataFrame):
        predictions = self.dl_recommender.predict(self._prepare_features(test_data))
        true_ratings = test_data['rating'].values

        # Existing metrics
        self.metrics['mae'] = np.mean(np.abs(predictions - true_ratings))
        self.metrics['rmse'] = np.sqrt(np.mean((predictions - true_ratings)**2))
        self.metrics['accuracy'] = np.mean((predictions.round() == true_ratings).astype(int))

        # New metrics for recommender systems
        k = 10  # Top-k recommendations to consider

        # Precision@k
        self.metrics['precision_at_k'] = self._precision_at_k(true_ratings, predictions, k)

        # Recall@k
        self.metrics['recall_at_k'] = self._recall_at_k(true_ratings, predictions, k)

        # F1-score@k
        self.metrics['f1_score_at_k'] = self._f1_score_at_k(true_ratings, predictions, k)

        # NDCG@k
        self.metrics['ndcg_at_k'] = self._ndcg_at_k(true_ratings, predictions, k)

        print(f"Accuracy: {self.metrics['accuracy']}")
        print(f"MAE: {self.metrics['mae']}")
        print(f"RMSE: {self.metrics['rmse']}")
        print(f"Precision@k: {self.metrics['precision_at_k']}")
        print(f"Recall@k: {self.metrics['recall_at_k']}")
        print(f"F1-score@k: {self.metrics['f1_score_at_k']}")
        print(f"NDCG@k: {self.metrics['ndcg_at_k']}")

    def _precision_at_k(self, true_ratings, predicted_ratings, k):
        # Implementation of Precision@k
        top_k_items = np.argsort(predicted_ratings)[::-1][:k]
        num_relevant = np.sum(true_ratings[top_k_items] >= self.relevance_threshold)
        return num_relevant / k

    def _recall_at_k(self, true_ratings, predicted_ratings, k):
        # Implementation of Recall@k
        top_k_items = np.argsort(predicted_ratings)[::-1][:k]
        num_relevant_recommended = np.sum(true_ratings[top_k_items] >= self.relevance_threshold)
        num_relevant_total = np.sum(true_ratings >= self.relevance_threshold)
        return num_relevant_recommended / num_relevant_total if num_relevant_total > 0 else 0

    def _f1_score_at_k(self, true_ratings, predicted_ratings, k):
        # Implementation of F1-score@k
        precision = self._precision_at_k(true_ratings, predicted_ratings, k)
        recall = self._recall_at_k(true_ratings, predicted_ratings, k)
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    def _ndcg_at_k(self, true_ratings, predicted_ratings, k):
        # Implementation of NDCG@k
        top_k_items = np.argsort(predicted_ratings)[::-1][:k]
        dcg = np.sum((2**true_ratings[top_k_items] - 1) / np.log2(np.arange(2, k + 2)))
        idcg = np.sum((2**np.sort(true_ratings)[::-1][:k] - 1) / np.log2(np.arange(2, k + 2)))
        return dcg / idcg if idcg > 0 else 0