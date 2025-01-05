from celery import shared_task
from django.core.cache import cache
from users.models import User, DietaryAssessment
from recipes.models import Recipe, Rating
from .model_ai import ImprovedHybridRecommender  # Changed from ai_model import HybridRecommender
import pandas as pd
import json
from typing import Dict, Any
import logging
from io import StringIO

logger = logging.getLogger(__name__)

def safe_json_load(data):
    """Safely load JSON data"""
    if isinstance(data, str):
        try:
            return json.loads(data)
        except json.JSONDecodeError:
            return data
    return data

@shared_task(bind=True, max_retries=3)
def fit_recommender_task(self) -> bool:
    """
    Task to fit the recommender model with retry capability
    """
    try:
        # Fetch all required data
        recipes_df = _fetch_recipes_data()
        users_df = _fetch_users_data()
        ratings_df = _fetch_ratings_data()
        
        # Debug prints
        print("Recipes DataFrame shape:", recipes_df.shape)
        print("Users DataFrame shape:", users_df.shape)
        print("Ratings DataFrame shape:", ratings_df.shape)
        
        # Merge data carefully with proper column handling
        merged_data = (ratings_df
            .merge(recipes_df, on='recipe_id', how='left', validate='m:1')
            .merge(users_df, on='user_id', how='left', validate='m:1'))
        
        # Handle missing values before training
        merged_data = merged_data.fillna({
            'rating': 0,
            'calories': 0,
            'protein': 0,
            'carbs': 0,
            'fat': 0,
            'fiber': 0
        })
        
        # Get the total number of recipes and users for the recommender
        num_recipes = len(recipes_df)
        num_users = len(users_df)
        
        try:
            # Initialize recommender with the correct parameters
            recommender = ImprovedHybridRecommender(num_recipes=num_recipes, num_users=num_users)
            
            # Train without immediate evaluation
            print("Starting model training...")
            recommender.train(recipes_df, merged_data, evaluate_after_training=False)
            print("Model training completed successfully!")
            
            # Cache the fitted recommender
            cache.set('hybrid_recommender', recommender, timeout=60*60*24)
            cache.set('recommender_fitted', True)
            cache.set('last_training_time', pd.Timestamp.now())
            
            # Instead of passing DataFrames directly, pass their JSON serialized versions
            evaluate_recommender.delay(
                merged_data.to_json(),
                users_df.to_json()
            )
            
            return True
            
        except Exception as train_exc:
            print(f"Error during model training: {str(train_exc)}")
            raise train_exc
            
    except Exception as exc:
        logger.error(f"Error fitting recommender: {exc}", exc_info=True)
        raise self.retry(exc=exc, countdown=300)  # Retry after 5 minutes

@shared_task
def evaluate_recommender(merged_data_json, users_df_json):
    """Separate task for evaluation with proper DataFrame handling"""
    try:
        # Deserialize the DataFrames using StringIO
        merged_data = pd.read_json(StringIO(merged_data_json))
        users_df = pd.read_json(StringIO(users_df_json))
        
        recommender = cache.get('hybrid_recommender')
        if recommender:
            metrics = recommender.evaluate_async(merged_data, users_df, max_users=100)
            cache.set('evaluation_metrics', metrics)
            return metrics
    except Exception as e:
        logger.error(f"Error in evaluation: {e}")
        return None

def _fetch_recipes_data() -> pd.DataFrame:
    """Efficiently fetch and prepare recipe data"""
    recipes_qs = Recipe.objects.select_related('nutrition').values(
        'recipe_id', 
        'name', 
        'ingredients', 
        'cuisine', 
        'recipe_info',
        'vegan', 
        'vegetarian', 
        'gluten_free', 
        'pescatarian', 
        'halal',
        'meal_type', 
        'dish_type', 
        'tags',
        'nutrition__calories', 
        'nutrition__protein', 
        'nutrition__carbs',
        'nutrition__fat', 
        'nutrition__fiber'
    )
    
    df = pd.DataFrame(list(recipes_qs))
    
    # Rename nutrition columns
    nutrition_mapping = {
        'nutrition__calories': 'calories',
        'nutrition__protein': 'protein',
        'nutrition__carbs': 'carbs',
        'nutrition__fat': 'fat',
        'nutrition__fiber': 'fiber'
    }
    df = df.rename(columns=nutrition_mapping)
    
    # Convert JSON strings to Python objects and handle missing values
    json_fields = ['ingredients', 'cuisine', 'tags', 'meal_type', 'dish_type']
    for field in json_fields:
        if field in df.columns:
            df[field] = df[field].apply(lambda x: 
                safe_json_load(x) if isinstance(x, str) else 
                ([] if x is None else x)
            )
    
    # Ensure boolean fields are properly set
    boolean_fields = ['gluten_free', 'halal', 'pescatarian', 'vegan', 'vegetarian']
    for field in boolean_fields:
        if field in df.columns:
            df[field] = df[field].fillna(False)
    
    # Fill missing numeric values
    numeric_fields = ['calories', 'protein', 'carbs', 'fat', 'fiber']
    for field in numeric_fields:
        if field in df.columns:
            df[field] = pd.to_numeric(df[field], errors='coerce').fillna(0)
    
    return df

def _fetch_users_data() -> pd.DataFrame:
    """Efficiently fetch and prepare user data"""
    users_with_assessments = User.objects.select_related('dietaryassessment').prefetch_related(
        'dietaryassessment__liked_ingredients',
        'dietaryassessment__disliked_ingredients'
    )
    
    users_data = []
    for user in users_with_assessments:
        user_data = {
            'user_id': user.id,
            'username': user.username,
            'weight': user.weight or 0,
            'height': user.height or 0,
            'age': user.age or 0,
            'gender': user.gender or '',
            'bmi': user.bmi or 0,
            'tdee': getattr(user, 'tdee', 0) or 0
        }
        
        assessment_data = _get_assessment_data(user)
        user_data.update(assessment_data)
        users_data.append(user_data)
    
    df = pd.DataFrame(users_data)
    
    # Ensure all columns have appropriate default values
    df['dietary_preferences'] = df.get('dietary_preferences', '[]').apply(safe_json_load)
    df['activity_levels'] = df.get('activity_levels', '[]').apply(safe_json_load)
    df['health_goals'] = df.get('health_goals', '[]').apply(safe_json_load)
    df['cuisine_preference']=df.get('cuisine_preference','[]').apply(safe_json_load)
    df['liked_ingredients'] = df.get('liked_ingredients', []).apply(lambda x: x if isinstance(x, list) else [])
    df['disliked_ingredients'] = df.get('disliked_ingredients', []).apply(lambda x: x if isinstance(x, list) else [])
    
    return df

def _get_assessment_data(user: User) -> Dict[str, Any]:
    """Extract dietary assessment data for a user"""
    if hasattr(user, 'dietaryassessment'):
        assessment = user.dietaryassessment
        return {
            'dietary_preferences': safe_json_load(assessment.dietary_preferences or '[]'),
            'activity_levels': safe_json_load(assessment.activity_levels or '[]'),
            'health_goals': safe_json_load(assessment.health_goals or '[]'),
            'cuisine_preference': safe_json_load(assessment.cuisine_preference or '[]'),
            'liked_ingredients': list(assessment.liked_ingredients.values_list('name', flat=True)),
            'disliked_ingredients': list(assessment.disliked_ingredients.values_list('name', flat=True)),
            'assessment': assessment.assessment or ''
        }
    return {
        'dietary_preferences': [],
        'activity_levels': [],
        'health_goals': [],
        'liked_ingredients': [],
        'disliked_ingredients': [],
        'cuisine_preference':[],
        'assessment': ''
    }

def _fetch_ratings_data() -> pd.DataFrame:
    """Fetch and prepare ratings data"""
    ratings_df = pd.DataFrame(
        Rating.objects.values('user_id', 'recipe_id', 'rating')
    )
    
    # Ensure rating is numeric and handle missing values
    ratings_df['rating'] = pd.to_numeric(ratings_df['rating'], errors='coerce').fillna(0)
    
    return ratings_df