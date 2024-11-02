# meal_planner/management/commands/generate_training_data.py
from django.core.management.base import BaseCommand
from django.contrib.auth import get_user_model
from users.models import DietaryAssessment
from recipes.models import Recipe, Rating,Ingredient
import pandas as pd
import numpy as np
import json
from faker import Faker
from datetime import datetime, timedelta
import random

User = get_user_model()
fake = Faker()

class Command(BaseCommand):
    help = 'Generate training data for the recommendation system'

    def add_arguments(self, parser):
        parser.add_argument('--users', type=int, default=1000,
                          help='Number of users to generate')
        parser.add_argument('--ratings-per-user', type=int, default=10,
                          help='Average number of ratings per user')

    def handle(self, *args, **options):
        num_users = options['users']
        ratings_per_user = options['ratings_per_user']

        self.stdout.write('Generating training data...')
        
        # Generate users and their dietary assessments
        self._generate_users(num_users)
        
        # Generate ratings
        self._generate_ratings(num_users, ratings_per_user)
        
        # Export to CSV for backup/analysis
        self._export_data()

    def _generate_users(self, num_users):
        self.stdout.write('Generating users and dietary assessments...')
        
        dietary_preferences = ['VEG', 'VER', 'GLU', 'HAL', 'PES', 'LOW_CARB', 'HIGH_PROTEIN']
        activity_levels = ['SED', 'LIG', 'MOD', 'LIG', 'VER']
        health_goals = ['LOS', 'GAI', 'MAI', 'MUS', 'FIT']
        
        users_data = []
        assessments_data = []
        
        for i in range(num_users):
            # Generate user data
            age = random.randint(18, 70)
            gender = random.choice(['Male', 'Female'])
            height = round(random.uniform(150, 200), 1)  # cm
            weight = round(random.uniform(45, 120), 1)  # kg
            
            # Calculate BMI and TDEE
            bmi = round(weight / ((height/100) ** 2), 1)
            activity_factor = random.uniform(1.2, 2.0)
            if gender == 'male':
                bmr = 88.362 + (13.397 * weight) + (4.799 * height) - (5.677 * age)
            else:
                bmr = 447.593 + (9.247 * weight) + (3.098 * height) - (4.330 * age)
            tdee = round(bmr * activity_factor)

            user = User.objects.create(
                username=f'user_{i}',
                email=f'user_{i}@example.com',
                age=age,
                gender=gender,
                height=height,
                weight=weight,
                bmi=bmi,
                tdee=tdee
            )
            users_data.append({
                'id': user.id,
                'username': user.username,
                'age': age,
                'gender': gender,
                'height': height,
                'weight': weight,
                'bmi': bmi,
                'tdee': tdee,
                'is_active':"True"
                
            })

            # Generate dietary assessment
            user_preferences = random.sample(dietary_preferences, k=random.randint(1, 3))
            user_activity = random.sample(activity_levels, k=random.randint(1, 2))
            user_goals = random.sample(health_goals, k=random.randint(1, 2))
            
            assessment = DietaryAssessment.objects.create(
                user=user,
                dietary_preferences=user_preferences,
                activity_levels=user_activity,
                health_goals=user_goals,
                tdee=tdee,
                bmi=bmi,
                assessment=f"Generated assessment for user {user.username}"
            )
            
            # Add random liked and disliked ingredients
            liked_ingredients = Ingredient.objects.order_by('?')[:random.randint(5, 15)]
            disliked_ingredients = Ingredient.objects.exclude(
                ingredients_id__in=[ing.ingredients_id for ing in liked_ingredients]
            ).order_by('?')[:random.randint(3, 10)]
            
            assessment.liked_ingredients.set(liked_ingredients)
            assessment.disliked_ingredients.set(disliked_ingredients)
            
            assessments_data.append({
                'user_id': user.id,
                'dietary_preferences': user_preferences,
                'activity_levels': user_activity,
                'health_goals': user_goals,
                'tdee': tdee,
                'bmi': bmi,
                'liked_ingredients': [ing.name for ing in liked_ingredients],
                'disliked_ingredients': [ing.name for ing in disliked_ingredients]
            })

        # Save to CSV
        pd.DataFrame(users_data).to_csv('meal_planner/training_data/users.csv', index=False)
        pd.DataFrame(assessments_data).to_csv('meal_planner/training_data/assessments.csv', index=False)

    def _generate_ratings(self, num_users, ratings_per_user):
        self.stdout.write('Generating ratings...')

        recipes = list(Recipe.objects.all())
        users = list(User.objects.all())
        ratings_data = []

        for user in users:
            # Get user's dietary assessment
            assessment = user.dietaryassessment
            liked_ingredients = set(assessment.liked_ingredients.values_list('name', flat=True))
            disliked_ingredients = set(assessment.disliked_ingredients.values_list('name', flat=True))

            # Generate ratings based on user preferences
            num_ratings = random.randint(
                int(ratings_per_user * 0.7),
                int(ratings_per_user * 1.3)
            )

            rated_recipes = set()  # Track rated recipes for this user

            for _ in range(num_ratings):
                recipe = random.choice(recipes)

                # Check if this user has already rated this recipe
                if Rating.objects.filter(user=user, recipe=recipe).exists():
                    continue  # Skip if this recipe has already been rated
                
                # Calculate rating based on user preferences
                base_rating = random.normalvariate(3.5, 0.5)  # Base rating between 1-5

                # Adjust rating based on ingredients
                recipe_ingredients = set(recipe.ingredients)  # Convert directly to a set
                liked_match = len(recipe_ingredients.intersection(liked_ingredients))
                disliked_match = len(recipe_ingredients.intersection(disliked_ingredients))

                rating = base_rating + (liked_match * 0.2) - (disliked_match * 0.3)
                rating = max(1, min(5, round(rating, 1)))  # Ensure rating is between 1-5

                # Create the rating only if it's unique
                Rating.objects.create(
                    user=user,
                    recipe=recipe,
                    rating=rating
                )

                ratings_data.append({
                    'user_id': user.id,
                    'recipe_id': recipe.recipe_id,
                    'rating': rating
                })

        # Save to CSV
        pd.DataFrame(ratings_data).to_csv('meal_planner/training_data/ratings.csv', index=False)


    def _export_data(self):
        self.stdout.write('Exporting data to CSV files...')
        
        # Export recipes
        recipes_data = []
        for recipe in Recipe.objects.select_related('nutrition').all():
            recipe_data = {
                'recipe_id': recipe.recipe_id,
                'name': recipe.name,
                'ingredients': recipe.ingredients,
                'cuisine': recipe.cuisine,
                'tags': recipe.tags,
                'meal_type': recipe.meal_type,
                'dish_type': recipe.dish_type,
                'calories': recipe.nutrition.calories,
                'protein': recipe.nutrition.protein,
                'carbs': recipe.nutrition.carbs,
                'fat': recipe.nutrition.fat,
                'fiber': recipe.nutrition.fiber,
                'vegetarian': recipe.vegetarian,
                'vegan': recipe.vegan,
                'gluten_free': recipe.gluten_free,
                'halal': recipe.halal,
                'pescatarian': recipe.pescatarian
            }
            recipes_data.append(recipe_data)
        
        pd.DataFrame(recipes_data).to_csv('meal_planner/training_data/recipes.csv', index=False)