# meal_planner/management/commands/load_training_data.py
from django.core.management.base import BaseCommand
import pandas as pd
from django.contrib.auth import get_user_model
from users.models import DietaryAssessment, Ingredient, Recipe, Rating
from django.db import transaction

User = get_user_model()

class Command(BaseCommand):
    help = 'Load training data from CSV files'

    def handle(self, *args, **options):
        try:
            with transaction.atomic():
                self._load_users()
                self._load_assessments()
                self._load_ratings()
            self.stdout.write(self.style.SUCCESS('Successfully loaded training data'))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'Error loading data: {str(e)}'))

    def _load_users(self):
        users_df = pd.read_csv('meal_planner/training_data/users.csv')
        for _, row in users_df.iterrows():
            User.objects.create(
                username=row['username'],
                email=row['email'],
                age=row['age'],
                gender=row['gender'],
                height=row['height'],
                weight=row['weight'],
                bmi=row['bmi'],
                tdee=row['tdee']
            )

    def _load_assessments(self):
        assessments_df = pd.read_csv('meal_planner/training_data/assessments.csv')
        for _, row in assessments_df.iterrows():
            user = User.objects.get(id=row['user_id'])
            assessment = DietaryAssessment.objects.create(
                user=user,
                dietary_preferences=row['dietary_preferences'],
                activity_levels=row['activity_levels'],
                health_goals=row['health_goals'],
                tdee=row['tdee'],
                bmi=row['bmi']
            )
            
            # Handle ingredients
            liked_ingredients = eval(row['liked_ingredients'])
            disliked_ingredients = eval(row['disliked_ingredients'])
            
            assessment.liked_ingredients.set(
                Ingredient.objects.filter(name__in=liked_ingredients)
            )
            assessment.disliked_ingredients.set(
                Ingredient.objects.filter(name__in=disliked_ingredients)
            )

    def _load_ratings(self):
        ratings_df = pd.read_csv('meal_planner/training_data/ratings.csv')
        ratings_to_create = []
        for _, row in ratings_df.iterrows():
            ratings_to_create.append(Rating(
                user_id=row['user_id'],
                recipe_id=row['recipe_id'],
                rating=row['rating']
            ))
        Rating.objects.bulk_create(ratings_to_create)