# In a file named management/commands/update_recommender.py in your app directory

from django.core.management.base import BaseCommand
from django.core.cache import cache
from users.models import Recipe, Rating, User
from users.ai_model import HybridRecommender

class Command(BaseCommand):
    help = 'Updates the HybridRecommender model with latest data'
    
    def safe_json_load(data):
        if isinstance(data, str):
            try:
                return json.loads(data)
            except json.JSONDecodeError:
                return data
        return data

    # Helper methods to prepare the data
    def _prepare_recipes(df):
        # Convert relevant columns to appropriate types
        numeric_columns = ['calories', 'protein', 'carbs', 'fat', 'fiber']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        # Handle list-like strings (e.g., for ingredients, tags, cuisine, meal_type, dish_type)
        list_columns = ['ingredients', 'tags', 'cuisine', 'meal_type', 'dish_type']
        for col in list_columns:
            df[col] = df[col].apply(lambda x: safe_json_load(x) if isinstance(x, str) else x)
            
        return df

    def _prepare_user_profiles(df):
        # Convert relevant columns to appropriate types
        df['tdee'] = pd.to_numeric(df['tdee'], errors='coerce')
        df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')
            
        # Ensure JSON fields are properly loaded
        json_fields = ['dietary_preferences', 'activity_levels', 'health_goals', 'goals']
        for field in json_fields:
            if field in df.columns:
                df[field] = df[field].apply(safe_json_load)
            
        return df

    def _prepare_user_interactions(df):
        # Convert rating to numeric
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
        return df

    def handle(self, *args, **options):
        # Convert QuerySets to DataFrames
        recipes_df = pd.DataFrame(Recipe.objects.select_related('nutrition').values(
            'recipe_id', 'name', 'ingredients', 'cuisine', 'recipe_info', 'vegan', 'vegetarian',
            'gluten_free', 'pescatarian', 'halal', 'meal_type', 'dish_type', 'tags',
            'nutrition__calories', 'nutrition__protein', 'nutrition__carbs', 'nutrition__fat', 'nutrition__fiber'
        ))
        ratings_df = pd.DataFrame(Rating.objects.all().values())
            
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
                'weight':user.weight,
                'height': user.height,
                'age': user.age,
                'gender': user.gender,
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
                    'goals': safe_json_load(assessment.goals),
                    'tdee': assessment.tdee,
                    'bmi': assessment.bmi,
                    'assessment': assessment.assessment
                })
            users_data.append(user_data)
            
        users_df = pd.DataFrame(users_data)

            # Rename nutrition fields in recipes_df
        recipes_df = recipes_df.rename(columns={
            'nutrition__calories': 'calories',
            'nutrition__protein': 'protein',
            'nutrition__carbs': 'carbs',
            'nutrition__fat': 'fat',
            'nutrition__fiber': 'fiber'
        })

        # Prepare user interactions
        user_interactions = ratings_df

        # Clean and prepare the data
        recipes = _prepare_recipes(recipes_df)
        user_profiles = _prepare_user_profiles(users_df)
        user_interactions = _prepare_user_interactions(user_interactions)
        
        recommender.fit(recipes, user_interactions, user_profiles)

        cache.set('hybrid_recommender', recommender, None)

        self.stdout.write(self.style.SUCCESS('Successfully updated recommender model'))