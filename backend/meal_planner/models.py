from django.db import models
from django.conf import settings
from django.utils import timezone
from django.contrib.postgres.fields import ArrayField
from django.core.validators import MaxLengthValidator,MinLengthValidator
from django.core.exceptions import ObjectDoesNotExist
import uuid
import string
from recipes.models import Recipe
import random
import logging
logger = logging.getLogger(__name__)

def content_file_path(instance, filename):
    return f'educational_content/{instance.content_id}/{filename}'

def generate_content_id():
    # Generate a random 8-character alphanumeric string
    chars = string.ascii_uppercase + string.ascii_lowercase + string.digits
    return ''.join(random.choice(chars) for _ in range(8))

class MealPlan(models.Model):
    DRAFT = 'DR'
    SAVED = 'SV'
    STATUS_CHOICES = [
        (DRAFT, 'Draft'),
        (SAVED, 'Saved'),
    ]

    meal_plan_id = models.CharField(max_length=8, default=generate_content_id, primary_key=True, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,  # This is the recommended way to reference User model
        on_delete=models.CASCADE,
        related_name='meal_plans'
    )
    name = models.CharField(max_length=255)
    description = models.TextField()
    meals_structure = models.JSONField(default=dict)
    meals = models.ManyToManyField(Recipe, related_name='meal_plan_recipes')
    tags = models.JSONField(default=dict)
    status = models.CharField(max_length=2, choices=STATUS_CHOICES, default=DRAFT)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    @property
    def total_calories(self):
        return sum(meal.calories for meal in self.meals.all())

    @property
    def total_protein(self):
        return sum(meal.protein for meal in self.meals.all())

    @property
    def total_carbs(self):
        return sum(meal.carbs for meal in self.meals.all())

    @property
    def total_fat(self):
        return sum(meal.fat for meal in self.meals.all())
    
    def save(self, *args, **kwargs):
        # Ensure meals_structure is properly serialized before saving
        if isinstance(self.meals_structure, dict):
            self.meals_structure = dict(self.meals_structure)
        super().save(*args, **kwargs)

    def calculate_nutritional_composition(self):
        """Calculate the nutritional composition of meals grouped by meal type."""
        nutritional_summary = {
            'breakfast': {"calories": 0, "protein": 0, "carbs": 0, "fat": 0},
            'lunch': {"calories": 0, "protein": 0, "carbs": 0, "fat": 0},
            'dinner': {"calories": 0, "protein": 0, "carbs": 0, "fat": 0},
            'snack': {"calories": 0, "protein": 0, "carbs": 0, "fat": 0}
        }

        try:
            # Loop through the meals_structure which contains the actual meal plan data
            for meal_type, recipes in self.meals_structure.items():
                if meal_type == 'summary':  # Skip the summary section
                    continue
                    
                # Each recipe in meals_structure now contains the nutritional info
                for recipe_data in recipes:
                    if isinstance(recipe_data, dict):
                        nutritional_summary[meal_type]["calories"] += int(recipe_data.get('calories', 0))
                        nutritional_summary[meal_type]["protein"] += int(recipe_data.get('protein', 0))
                        nutritional_summary[meal_type]["carbs"] += int(recipe_data.get('carbs', 0))
                        nutritional_summary[meal_type]["fat"] += int(recipe_data.get('fat', 0))

            logger.info(f"Calculated nutritional summary: {nutritional_summary}")
            return nutritional_summary
            
        except Exception as e:
            logger.error(f"Error calculating nutritional composition: {str(e)}")
            logger.error(f"meals_structure content: {self.meals_structure}")
            return nutritional_summary
