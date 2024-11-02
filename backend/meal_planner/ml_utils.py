import pandas as pd
import json
from django.core.cache import cache
from users.models import User, DietaryAssessment, Ingredient
from recipes.models import Recipe, Rating
from .ai_model import HybridRecommender

def safe_json_load(data):
    if isinstance(data, str):
        try:
            return json.loads(data)
        except json.JSONDecodeError:
            return data
    return data

def _prepare_recipes(df):
    # Convert relevant columns to appropriate types
    numeric_columns = ['calories', 'protein', 'carbs', 'fat', 'fiber']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    # Handle list-like strings (e.g., for ingredients, tags, cuisine, meal_type, dish_type)
    list_columns = ['ingredients', 'tags', 'cuisine', 'meal_type', 'dish_type']
    for col in list_columns:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: safe_json_load(x) if isinstance(x, str) else x)
            
    return df

def _prepare_user_profiles(df):
    # Convert relevant columns to appropriate types if they exist
    numeric_columns = ['tdee', 'bmi', 'weight', 'height', 'age']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    # Ensure JSON fields are properly loaded if they exist
    json_fields = ['dietary_preferences', 'activity_levels', 'health_goals', 'goals']
    for field in json_fields:
        if field in df.columns:
            df[field] = df[field].apply(safe_json_load)
            
    return df

def _prepare_user_interactions(df):
    # Convert rating to numeric if the column exists
    if 'rating' in df.columns:
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    else:
        print("Warning: 'rating' column not found in user interactions data.")
    return df

def fit_recommender():
    # Your existing fit_recommender function
    # Convert QuerySets to DataFrames
    recipe_count = Recipe.objects.count()
    print(f"Debug: Total number of recipes in the database: {recipe_count}")
    recipes_qf = Recipe.objects.select_related('nutrition').values(
        'recipe_id', 'name', 'ingredients', 'cuisine', 'recipe_info', 'vegan', 'vegetarian',
        'gluten_free', 'pescatarian', 'halal', 'meal_type', 'dish_type', 'tags',
        'nutrition__calories', 'nutrition__protein', 'nutrition__carbs', 'nutrition__fat', 'nutrition__fiber'
    )
    print(f"Debug: Recipe queryset SQL: {recipes_qf.query}")
    
    recipes_df = pd.DataFrame(list(recipes_qf))
    print(f"Debug: Recipes DataFrame shape after query: {recipes_df.shape}")
    print(f"Debug: Recipes DataFrame columns: {recipes_df.columns.tolist()}")
    # Rename nutrition fields in recipes_df
    recipes_df = recipes_df.rename(columns={
        'nutrition__calories': 'calories',
        'nutrition__protein': 'protein',
        'nutrition__carbs': 'carbs',
        'nutrition__fat': 'fat',
        'nutrition__fiber': 'fiber'
    })
    ratings_df = pd.DataFrame(Rating.objects.all().values('user_id', 'recipe_id', 'rating'))
            
    # Fetch users with their dietary assessments
    users_with_assessments = User.objects.select_related('dietaryassessment').prefetch_related(
        'dietaryassessment__liked_ingredients',
        'dietaryassessment__disliked_ingredients'
    )
            
    users_data = []
    for user in users_with_assessments:
        user_data = {
            'user_id': user.id,
            'username': user.username,
            'weight': user.weight,
            'height': user.height,
            'age': user.age,
            'gender': user.gender,
            'bmi':user.bmi
            # Add other User fields as needed
        }
        if hasattr(user, 'dietaryassessment'):
            assessment = user.dietaryassessment
            user_data.update({
                'dietary_preferences': safe_json_load(assessment.dietary_preferences),
                'activity_levels': safe_json_load(assessment.activity_levels),
                'health_goals': safe_json_load(assessment.health_goals),
                'liked_ingredients': list(assessment.liked_ingredients.values_list('name', flat=True)),
                'disliked_ingredients': list(assessment.disliked_ingredients.values_list('name', flat=True)),
                'tdee': assessment.tdee,
                'assessment': assessment.assessment
            })
        users_data.append(user_data)
            
    users_df = pd.DataFrame(users_data)

    

    # Prepare user interactions
    user_interactions = ratings_df

    # Clean and prepare the data
    recipes = _prepare_recipes(recipes_df)
    user_profiles = _prepare_user_profiles(users_df)
    user_interactions = _prepare_user_interactions(user_interactions)


    recommender = HybridRecommender()
    recommender.fit(recipes, user_interactions, user_profiles)

    # Store the fitted model in cache
    cache.set('hybrid_recommender', recommender, timeout=60*60*24)  # 24 hour expiration