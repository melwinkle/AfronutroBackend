from django.test import TestCase
from django.urls import reverse
from rest_framework.test import APIClient
from rest_framework import status
from rest_framework.authtoken.models import Token
from django.contrib.auth import get_user_model
from users.models import DietaryAssessment, DietaryPreference,ActivityLevel,HealthGoal
from recipes.models import Recipe, Rating Favorite, NutritionalInformation, Ingredient
from meal_planner.models import MealPlan
from meal_planner.serializers import MealPlanSerializer
from recipes.serializers import RecipeSerializer
from users.serializers import DietaryAssessmentSerializer
from django.core.cache import cache
from unittest.mock import patch
from django.db import connection
import json
from django.core.files.uploadedfile import SimpleUploadedFile
# Create your tests here.
# Move MockRecommender outside of setUp
class MockRecommender:
    def get_recommendations(self, user_profile):
        return {'breakfast': ['Test Recipe 1'], 'lunch': ['Test Recipe 2']}
    
class MealPlanViewsTestCase(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.register_url = 'http://localhost:8000/register/'
        self.login_url = 'http://localhost:8000/login/'
        self.meal_plan='http://localhost:8000/meals/meal-plans/generate/'
        self.meal_plans='http://localhost:8000/meals/meal-plans/'
        self.meal_plans_url='http://localhost:8000/meals/meal-plans/{}/save/'
        self.meal_plans_urls='http://localhost:8000/meals/meal-plans/{}/customize/'
        self.meal_plans_ns='http://localhost:8000/meals/meal-plans/{}/nutritional-summary/'
        self.rating_url='http://localhost:8000/recipes/rating/{}/'
        # User registration data
        self.user_data = {
            "email": "testuser3@example.com",
            "username": "User3",
            "password": "password123$",
            "password2": "password123$",
            "age": 24,
            "gender": "Male",
            "height": 157.0,
            "weight": 68,
            "is_verified": False,
            "tdee":1242,
            "activity_level": 1.5,
            "bmi":25.6
        }
        
        # Register and activate the user
        response = self.client.post(self.register_url, self.user_data, format='json')
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        
        self.user = get_user_model().objects.get(email=self.user_data['email'])
        self.user.is_active = True
        self.user.save()
        
        # Login the user
        login_data = {
            'email': self.user_data['email'],
            'password': self.user_data['password']
        }
        login_response = self.client.post(self.login_url, login_data, format='json')
        self.assertEqual(login_response.status_code, status.HTTP_200_OK)
        self.assertIn('token', login_response.data)
        
        # Set up authentication for further requests
        self.token = login_response.data['token']
        self.client.credentials(HTTP_AUTHORIZATION=f'Token {self.token}')

        # Create a dietary assessment for the user
        self.dietary_assessment = DietaryAssessment.objects.create(
            user=self.user,
            activity_levels=json.dumps(['MOD']),
            tdee=2000,
            bmi=22.5,
            health_goals=json.dumps(['LOS']),
            dietary_preferences=json.dumps(['VEG'])
        )
        # Now assign the ingredients to the many-to-many fields
        self.ingredient = Ingredient.objects.create(name='Chicken',calories=234,protein=23,fat=13,carbs=2,minerals=["Calcium"],vitamins=["Vitamin B1"])
        self.ingredient1 = Ingredient.objects.create(name='Broccoli',calories=234,protein=23,fat=13,carbs=2,minerals=["Calcium"],vitamins=["Vitamin B1"])
        self.ingredient2 =  Ingredient.objects.create(name='Mushroom',calories=234,protein=23,fat=13,carbs=2,minerals=["Calcium"],vitamins=["Vitamin B2"])
        
        self.ingredient.substitutes.set([])
        self.ingredient1.substitutes.set([])
        self.ingredient2.substitutes.set([])
        
        self.dietary_assessment.liked_ingredients.set([self.ingredient, self.ingredient1])
        self.dietary_assessment.disliked_ingredients.set([self.ingredient2])


        # Create recipes
        self.create_test_recipes()
        
        data = {'rating': 4,'comment':"Wow!"}
        rating_response = self.client.post(self.rating_url.format(self.recipe1.recipe_id), data, format='json')
        if rating_response.status_code != status.HTTP_201_CREATED:
            print(f"Error response content rating: {rating_response.content}")
        self.assertEqual(rating_response.status_code, status.HTTP_201_CREATED)
        
        data1 = {'rating': 5,'comment':"Wow!"}
        ratingresponse = self.client.post(self.rating_url.format(self.recipe2.recipe_id), data, format='json')
        if ratingresponse.status_code != status.HTTP_201_CREATED:
            print(f"Error response content rating: {ratingresponse.content}")
        self.assertEqual(ratingresponse.status_code, status.HTTP_201_CREATED)

        # Use MockRecommender
        self.mock_recommender = MockRecommender()
        # Instead of caching the MockRecommender instance, mock the cache.get method
        cache.get = lambda key: self.mock_recommender if key == 'hybrid_recommender' else None
    def create_test_recipes(self):
        # Create nutritional information
        self.nutritional_info1 = NutritionalInformation.objects.create(
            calories=200, protein=10, fat=5, carbs=30, fiber=34
        )
        self.nutritional_info2 = NutritionalInformation.objects.create(
            calories=300, protein=15, fat=10, carbs=25, fiber=28
        )

        # Create recipes
        self.recipe1 = Recipe.objects.create(
            name='Test Recipe 1',
            ingredients=['ingredient1', 'ingredient2'],
            cuisine=['ghanaian'],
            recipe_info='A test recipe for unit testing',
            vegan=True,
            vegetarian=True,
            gluten_free=False,
            pescatarian=False,
            halal=True,
            meal_type=['dinner'],
            dish_type=['main'],
            tags=['high-protein'],
            nutrition=self.nutritional_info1
        )

        self.recipe2 = Recipe.objects.create(
            name='Test Recipe 2',
            ingredients=['ingredient3', 'ingredient4'],
            cuisine=['italian'],
            recipe_info='Another test recipe',
            vegan=False,
            vegetarian=True,
            gluten_free=True,
            pescatarian=False,
            halal=True,
            meal_type=['lunch'],
            dish_type=['side'],
            tags=['low-carb'],
            nutrition=self.nutritional_info2
        )

    @patch('users.views.cache.get')
    @patch('users.views.cache.set')
    def test_generate_meal_plan(self, mock_cache_set, mock_cache_get):
        mock_cache_get.return_value = None  # Simulate cache miss

        response = self.client.post(self.meal_plan)
        
        if response.status_code != status.HTTP_201_CREATED:
            print(f"Error response content CREATE: {response.content}")
        
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        
        # Check response content
        self.assertIn('name', response.data)
        self.assertEqual(response.data['name'], "AI Generated Meal Plan")
        
        self.assertIn('meal_plan_id', response.data)
        self.assertIsNotNone(response.data['meal_plan_id'])
        
        # Verify that a MealPlan was created in the database
        self.assertEqual(MealPlan.objects.count(), 1)
        
        # Verify that the meal plan was cached
        self.assertTrue(mock_cache_set.called)
        
        # Store the created meal plan ID for use in other tests
        self.draft_meal_plan_id = response.data['meal_plan_id']

        # Verify the cached meal plan
        cached_meal_plan = mock_cache_set.call_args[0][1]  # Get the second argument of the first call
        self.assertIsInstance(cached_meal_plan, MealPlan)
        self.assertEqual(cached_meal_plan.meal_plan_id, self.draft_meal_plan_id)


    def test_get_meal_plans(self):
        MealPlan.objects.create(user=self.user, name="Test Meal Plan 1", status=MealPlan.SAVED)
        MealPlan.objects.create(user=self.user, name="Test Meal Plan 2", status=MealPlan.SAVED)
        
        response = self.client.get(self.meal_plans)
        if response.status_code != status.HTTP_200_OK:
            print(f"Error response content GET: {response.content}")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data), 2)

    @patch('users.views.cache.get')
    def test_save_meal_plan(self,mock_cache_get):
        # Create a draft meal plan
        draft_meal_plan = MealPlan.objects.create(
            user=self.user,
            name="Draft Meal Plan",
            status=MealPlan.DRAFT
        )
        
        # Mock the cache to return our draft meal plan
        mock_cache_get.return_value = draft_meal_plan

        response = self.client.post(self.meal_plans_url.format(draft_meal_plan.meal_plan_id))
        if response.status_code != status.HTTP_200_OK:
            print(f"Error response content SAVE: {response.content}")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        
    def test_customize_meal_plan(self):
        meal_plan = MealPlan.objects.create(user=self.user, name="Test Meal Plan", status=MealPlan.DRAFT)
        cache.set(f'meal_plan_draft_{meal_plan.meal_plan_id}', meal_plan)

        data = {'name': 'Updated Meal Plan'}
        response = self.client.put(self.meal_plans_urls.format(meal_plan.meal_plan_id), data)
        if response.status_code != status.HTTP_200_OK:
            print(f"Error response content CS: {response.content}")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['name'], 'Updated Meal Plan')

    def test_get_nutritional_summary(self):
        meal_plan = MealPlan.objects.create(user=self.user, name="Test Meal Plan")
        meal_plan.meals.add(self.recipe1, self.recipe2)

        url = f'/meal-plans/{meal_plan.meal_plan_id}/nutritional-summary/'
        response = self.client.get(self.meal_plans_ns.format(meal_plan.meal_plan_id))
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('total_calories', response.data)
        self.assertIn('total_protein', response.data)
        self.assertIn('total_carbs', response.data)
        self.assertIn('total_fat', response.data)
        self.assertIn('tdee', response.data)
        self.assertIn('calorie_difference', response.data)
        
        expected_calories = self.nutritional_info1.calories + self.nutritional_info2.calories
        expected_protein = self.nutritional_info1.protein + self.nutritional_info2.protein
        expected_carbs = self.nutritional_info1.carbs + self.nutritional_info2.carbs
        expected_fat = self.nutritional_info1.fat + self.nutritional_info2.fat
        
        self.assertEqual(response.data['total_calories'], expected_calories)
        self.assertEqual(response.data['total_protein'], expected_protein)
        self.assertEqual(response.data['total_carbs'], expected_carbs)
        self.assertEqual(response.data['total_fat'], expected_fat)
        self.assertEqual(response.data['tdee'], 2000)
        self.assertEqual(response.data['calorie_difference'], expected_calories - 2000)

    def tearDown(self):
        cache.clear()