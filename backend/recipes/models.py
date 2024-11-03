from django.db import models
from django.conf import settings
from django.utils import timezone
from django.contrib.postgres.fields import ArrayField
from django.core.validators import MaxLengthValidator,MinLengthValidator
from django.core.exceptions import ObjectDoesNotExist
from django.utils import timezone
import uuid
import string
import random
import logging
logger = logging.getLogger(__name__)


def generate_content_id():
    # Generate a random 8-character alphanumeric string
    chars = string.ascii_uppercase + string.ascii_lowercase + string.digits
    return ''.join(random.choice(chars) for _ in range(8))

class NutritionalInformation(models.Model):
    nutrition_info_id = models.CharField(max_length=8, primary_key=True, default=generate_content_id, editable=False)
    calories = models.IntegerField(default=0.0)
    protein = models.FloatField(default=0.0)
    carbs = models.FloatField(default=0.0)
    fat = models.FloatField(default=0.0)
    fiber=models.FloatField(default=0.0)
    
def recipe_file_path(instance, filename):
    return f'recipe/{instance.content_id}/{filename}'

class TagsType(models.TextChoices):
        DAIRY_FREE = 'dairy-free', 'Dairy-Free'
        KETO = 'keto', 'Keto'
        PALEO = 'paleo','Paleo'
        HIGH_PROTEIN = 'high-protein','High-Protein'
        LOW_PROTEIN = 'low-protein','Low-Protein'
        LOW_CARB = 'low-carb', 'Low-Carb'
        NUT_FREE = 'nut-free', 'Nut_Free'
        SHELLFISH_FREE = 'shellfish-free', 'Shellfish-Free'
        LACTOSE_FREE = 'lactose-free', 'Lactose-Free'
        EGG_FREE = 'egg-free', 'Egg-Free'
        PEANUT_FREE = 'peanut-free', 'Peanut-Free'
        SOY_FREE = 'soy_free', 'Soy-Free'
        LOW_SUGAR = 'low-sugar', 'Low-Sugar'
        SPICY = 'spicy','Spicy'
        SWEET = 'sweet', 'Sweet'
        SAVORY = 'savory','Savory'
        ORGANIC = 'organic','Organic'
        HIGH_FIBER = 'high-fiber','High-Fiber'
        HIGH_FAT= 'high-fat','High-Fat',
        MODERATE_CARBS='moderate-carbs','Moderate-Carbs'
        LOW_FAT='low-fat','Low-Fat'
        MODERATE_PROTEIN='moderate-protein','Moderate-Protein'
        BALANCED='balanced','Balanced'
class MealType(models.TextChoices):
        BREAKFAST = 'breakfast', 'Breakfast'
        LUNCH = 'lunch', 'Lunch'
        DINNER = 'dinner', 'Dinner'
        SNACK = 'snack', 'Snack'

class DishType(models.TextChoices):
        MAIN = 'main', 'Main'
        SIDE = 'side', 'Side'
        PROTEIN = 'protein', 'Protein'
        VEGETABLE = 'vegetable', 'Vegetable'
        CARB = 'carb', 'Carb'
        DESSERT = 'dessert', 'Dessert'
        APPETIZER = 'appetizer', 'Appetizer'
        SOUP = 'soup', 'Soup'
        SALAD = 'salad', 'Salad'
        SAUCE = 'sauce', 'Sauce'
        DRINK = 'drink', 'Drink'
        ONE_POT ='one-pot','One-Pot'
    
class CuisineType(models.TextChoices):
        GHANAIAN = 'ghanaian', 'Ghanaian'
        NIGERIAN = 'nigerian', 'Nigeria'
        WESTAFRICAN = 'west african', 'West African'
        EUROPEAN = 'european', 'European'
        AMERICAN = 'american', 'American'
        ASIAN = 'asian', 'Asian'
        MIDDLEEASTERN = 'middle eastern', 'Middle Eastern'
        INDIAN = 'indian', 'Indian'
        CHINESE = 'chinese', 'Chinese'
        JAPANESE = 'japanese', 'Japanese'
        KOREAN = 'korean', 'Korean'
        THAI = 'thai', 'Thai'
        VIETNAMESE = 'vietnamese', 'Vietnamese'
        ITALIAN = 'italian', 'Italian'
        MEXICAN = 'mexican', 'Mexican'
        SPANISH = 'spanish', 'Spanish'
        FRENCH = 'french', 'French'
        GERMAN = 'german', 'German'
        BRITISH = 'british', 'British'
        AUSTRALIAN = 'australian', 'Australian'
        CANADIAN = 'canadian', 'Canadian'
             
class Recipe(models.Model):
    
    
    recipe_id = models.CharField(max_length=8, primary_key=True, default=generate_content_id, editable=False)
    name = models.CharField(max_length=200)
    ingredients = ArrayField(models.CharField(max_length=100))
    cuisine = ArrayField(
        models.CharField(max_length=20, choices=CuisineType.choices),
        size=5,  # Limit to 5 dish types per recipe
    )
    recipe_info = models.TextField()
    
    # Dietary restrictions
    vegan = models.BooleanField(default=False)
    vegetarian = models.BooleanField(default=False)
    gluten_free = models.BooleanField(default=False)
    pescatarian = models.BooleanField(default=False)
    halal = models.BooleanField(default=False)
    
    meal_type = ArrayField(
        models.CharField(max_length=20, choices=MealType.choices),
        size=2,  # Limit to 3 dish types per recipe
    )
    dish_type = ArrayField(
        models.CharField(max_length=20, choices=DishType.choices),
        size=3,  # Limit to 3 dish types per recipe
    )
    tags = ArrayField(models.CharField(max_length=50,choices=TagsType.choices,), size=5,)
    nutrition = models.OneToOneField(NutritionalInformation, on_delete=models.CASCADE)
    duration = models.IntegerField(null=True,default=10)
    image= models.FileField(upload_to=recipe_file_path, blank=True, null=True)

    def __str__(self):
        return self.name



class Rating(models.Model):
    rating_id=models.CharField(max_length=8, primary_key=True, default=generate_content_id, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,  # This is the recommended way to reference User model
        on_delete=models.CASCADE,
        related_name='recipes'
    )
    recipe = models.ForeignKey(Recipe, on_delete=models.CASCADE)
    rating = models.FloatField(default=0.0)
    comment = models.TextField(blank=True)
    timestamp = models.DateTimeField(default=timezone.now)

    class Meta:
        unique_together = ('user', 'recipe')  # Ensures a user can rate a recipe only once

    def __str__(self):
        return f'{self.user} rated {self.recipe} - {self.rating}'
    
class Favorite(models.Model):
    favorite_id=models.CharField(max_length=8, primary_key=True, default=generate_content_id, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,  # This is the recommended way to reference User model
        on_delete=models.CASCADE,
        related_name='recipes_favorites'
    )
    recipe = models.ForeignKey(Recipe, on_delete=models.CASCADE)
    added_on = models.DateTimeField(default=timezone.now)

    class Meta:
        unique_together = ('user', 'recipe')  # Ensures a user can favorite a recipe only once

    def __str__(self):
        return f'{self.user} favorited {self.recipe}'

class Ingredient(models.Model):
    ingredients_id=models.CharField(max_length=8, primary_key=True, default=generate_content_id, editable=False)
    name = models.CharField(max_length=100, unique=True)
    calories = models.FloatField()
    carbs = models.FloatField()
    protein = models.FloatField()
    fat = models.FloatField()
    minerals = models.JSONField()
    vitamins = models.JSONField()
    substitutes = models.ManyToManyField('self', symmetrical=False, blank=True)

    def __str__(self):
        return self.name