from django.test import TestCase
from django.urls import reverse
from rest_framework.test import APIClient
from rest_framework import status
from rest_framework.authtoken.models import Token
from django.contrib.auth import get_user_model
from .models import Recipe, DietaryAssessment, MealPlan, Ingredient, Rating, Favorite, EducationalContent, NutritionalInformation, DietaryPreference,ActivityLevel,HealthGoal
from .serializers import RecipeSerializer, DietaryAssessmentSerializer, MealPlanSerializer
from django.core.cache import cache
from unittest.mock import patch
from django.db import connection
import json
from django.core.files.uploadedfile import SimpleUploadedFile


User = get_user_model()
print(f"Test database name: {connection.settings_dict['NAME']}")

class UserAPITestCase(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.register_url = 'http://localhost:8000/register/'  # Adjust URL for Node.js
        self.login_url = 'http://localhost:8000/login/'
        self.logout_url = 'http://localhost:8000/logout/'

        self.user_data={
            "email": "testuser@example.com",
            "username": "User",
            "password": "password123$",
            "password2": "password123$",
            "age": 24,
            "gender": "Female",
            "height": 157.0,
            "weight": 68,
            "is_verified": False,  
        }
        

    def test_user_registration(self):

        response = self.client.post(self.register_url, self.user_data, format='json')
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        created_user = get_user_model().objects.get(email=self.user_data['email'])
        self.assertEqual(created_user.username, self.user_data['username'])
        self.assertEqual(created_user.age, self.user_data['age'])
        self.assertEqual(created_user.gender, self.user_data['gender'])
        self.assertEqual(created_user.height, self.user_data['height'])
        self.assertEqual(created_user.weight, self.user_data['weight'])



    def test_user_login(self):
        # Register the user first
        self.client.post(self.register_url, self.user_data, format='json')
        # Manually activate the user
        user = User.objects.get(email=self.user_data['email'])
        user.is_active = True
        user.save()

        # Login the user
        login_data = {
            'email': self.user_data['email'],
            'password': self.user_data['password']
        }
        login_response = self.client.post(self.login_url, login_data, format='json')

        # Assert login was successful
        self.assertEqual(login_response.status_code, status.HTTP_200_OK)
        self.assertIn('token', login_response.data)
        self.assertIn('user_id', login_response.data)
        self.assertIn('email', login_response.data)

        # Store the token for further testing (e.g., logout)
        self.token = login_response.data['token']

    def test_user_logout(self):
        """
        Test successful user logout after logging in.
        """
        # Register and log in the user first
        self.test_user_login()

        # Set token in the header
        self.client.credentials(HTTP_AUTHORIZATION='Token ' + self.token)

        # Logout the user
        logout_response = self.client.post(self.logout_url)

        # Assert logout was successful
        self.assertEqual(logout_response.status_code, status.HTTP_200_OK)

    def test_invalid_registration(self):
        """
        Test invalid registration with an already registered email.
        """
        # Register the user first
        self.client.post(self.register_url, self.user_data, format='json')

        # Attempt to register with the same email
        invalid_register_data = {
            'email': self.user_data['email'],
            'username': self.user_data['username'],
            'password': 'newpassword$',
            'password2': "newpassword$",
            'age': 24,
            'gender': 'Male',
            'height': 157.0,
            'weight': 68,
            'is_verified': False, 
        }
        invalid_register_response = self.client.post(self.register_url, invalid_register_data, format='json')

        # Assert registration with the same email fails
        self.assertEqual(invalid_register_response.status_code, status.HTTP_400_BAD_REQUEST)
        # Check if the error messages match what you expect
        self.assertEqual(invalid_register_response.data['email'][0].code, 'unique')
        self.assertEqual(invalid_register_response.data['username'][0].code, 'unique')




class RecipeTests(TestCase):
    def setUp(self):
        # URLs for the API endpoints
        self.recipe_list_url = "http://localhost:8000/recipes/"  # Assuming the URL name is 'recipe-list'
        self.recipe_detail_url = "http://localhost:8000/recipes/{}/"  # Assuming the URL name is 'recipe-detail'

        # Create nutritional information for the recipe
        self.nutritional_info = NutritionalInformation.objects.create(
            calories=200,  
            protein=10,
            fat=5,
            carbs=30,
            fiber=34
        )

        # Valid recipe data
        self.valid_recipe_data = {
            'name': 'Test Recipe',
            'ingredients': ['ingredient1', 'ingredient2'],
            'cuisine': ['italian'],  # Ensure the cuisine matches your model's choices
            'recipe_info': 'A test recipe for unit testing',
            'vegan': True,
            'vegetarian': True,
            'gluten_free': False,
            'pescatarian': False,
            'halal': True,
            'meal_type': ['dinner'],  # Ensure meal type matches your model's choices
            'dish_type': ['main'],  # Ensure dish type matches your model's choices
            'tags': ['high-protein'],
            'nutrition': {
                'calories': 200,
                'protein': 10,
                'fat': 5,
                'carbs': 30,
                'fiber': 34
            }
        }

        # Create a recipe for update and delete tests
        nutrition = NutritionalInformation.objects.create(**self.valid_recipe_data['nutrition'])
        recipe_data = self.valid_recipe_data.copy()
        recipe_data['nutrition'] = nutrition
        self.recipe = Recipe.objects.create(**recipe_data)

    def test_create_recipe(self):
        """
        Test creating a new recipe.
        """
        response = self.client.post(self.recipe_list_url, data=json.dumps(self.valid_recipe_data),content_type='application/json')
        
        # Assert that the recipe was created successfully
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertIn('message', response.data)
        self.assertIn('recipe', response.data)

    def test_get_all_recipes(self):
        """
        Test retrieving all recipes.
        """
        response = self.client.get(self.recipe_list_url)
        
        # Assert that the response contains the recipes
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIsInstance(response.data, list)  # Check if the response is a list
        self.assertGreater(len(response.data), 0)  # Check that at least one recipe is present

    def test_get_recipe_detail(self):
        """
        Test retrieving a specific recipe by ID.
        """
        response = self.client.get(self.recipe_detail_url.format(self.recipe.recipe_id))  # Use the created recipe's ID
        
        # Assert that the response contains the correct recipe
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['name'], self.recipe.name)
        self.assertEqual(response.data['recipe_info'], self.recipe.recipe_info)

    def test_update_recipe(self):
        """
        Test updating an existing recipe.
        """
        update_data = {
            'name': 'Updated Test Recipe',
            'recipe_info': 'An updated test recipe for unit testing',
            # Include other fields you want to update
        }
        
        response = self.client.put(self.recipe_detail_url.format(self.recipe.recipe_id), update_data, format='json',content_type='application/json')

        # Assert that the recipe was updated successfully
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('message', response.data)
        self.assertEqual(response.data['recipe']['name'], update_data['name'])

        # Verify that the recipe was actually updated in the database
        self.recipe.refresh_from_db()
        self.assertEqual(self.recipe.name, update_data['name'])

    def test_delete_recipe(self):
        """
        Test deleting an existing recipe.
        """
        response = self.client.delete(reverse('recipe-detail', args=[self.recipe.recipe_id]))

        # Assert that the recipe was deleted successfully
        self.assertEqual(response.status_code, status.HTTP_204_NO_CONTENT)

        # Verify that the recipe no longer exists in the database
        self.assertFalse(Recipe.objects.filter(recipe_id=self.recipe.recipe_id).exists())


class DietaryAssessmentViewTestCase(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.list_create_url = 'http://localhost:8000/dietary-assessment/'
        self.register_url='http://localhost:8000/register/'
        self.login_url='http://localhost:8000/login/'
       
        
        # User registration data
        self.user_data = {
            "email": "testuser2@example.com",
            "username": "User2",
            "password": "password123$",
            "password2": "password123$",
            "age": 24,
            "gender": "Male",
            "height": 157.0,
            "weight": 68,
            "is_verified": False,
        }
        
        # Register and log in the user
        response = self.client.post(self.register_url, self.user_data, format='json')
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)

        
        # Activate the user
        self.user = get_user_model().objects.get(email=self.user_data['email'])
        self.client.force_authenticate(user=self.user)
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
 

        # Store the token and set up authentication for further requests
        self.token = login_response.data['token']
        self.client.credentials(HTTP_AUTHORIZATION=f'Token {self.token}')
        
        # Create test ingredients
        self.ingredient1 = Ingredient.objects.create(name='Broccoli',calories=234,protein=23,fat=13,carbs=2,minerals=["Calcium"],vitamins=["Vitamin B1"])
        self.ingredient2 = Ingredient.objects.create(name='Chicken',calories=254,protein=33,fat=23,carbs=4,minerals=["Calcium"],vitamins=["Vitamin B1"])
        self.ingredient3 = Ingredient.objects.create(name='Tomato',calories=264,protein=12,fat=13,carbs=6,minerals=["Calcium"],vitamins=["Vitamin B1"])

        self.ingredient1.substitutes.set([])
        self.ingredient2.substitutes.set([])
        self.ingredient3.substitutes.set([])
        # Set up test data
        self.valid_payload = {
            'dietary_preferences': ['VGT'],
            'activity_levels': ['MOD'],
            'health_goals': ['LOS'],
            'liked_ingredients': ['Broccoli', 'Tomato'],
            'disliked_ingredients': ['Chicken'],
            'goals': ['Lose 10 pounds in 3 months']
        }

        

        

    def test_get_questionnaire_and_ingredients(self):
        response = self.client.get(self.list_create_url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        
        # Check if the response contains the questionnaire and ingredients
        self.assertIn('questionnaire', response.data)
        self.assertIn('ingredients', response.data)
        
        # Check if the questionnaire contains the expected fields
        questionnaire = response.data['questionnaire']
        self.assertIn('dietary_preferences', questionnaire)
        self.assertIn('activity_levels', questionnaire)
        self.assertIn('health_goals', questionnaire)
        
        # Check if all ingredients are returned
        ingredients = response.data['ingredients']
        self.assertEqual(len(ingredients), Ingredient.objects.count())
        
        # Check if the ingredients contain the expected fields
        for ingredient in ingredients:
            self.assertIn('ingredients_id', ingredient)
            self.assertIn('name', ingredient)

    def test_create_dietary_assessment(self):
        # Create a new dietary assessment
        # Ensure no dietary assessment exists for the user
        self.assertEqual(DietaryAssessment.objects.filter(user=self.user).count(), 0)

        # Create a new dietary assessment
        response = self.client.post(self.list_create_url, self.valid_payload, format='json')
        
        # Print response content if there's an error
        if response.status_code != status.HTTP_201_CREATED:
            print(f"Error response content: {response.content}")

        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(DietaryAssessment.objects.filter(user=self.user).count(), 1)

        # Verify the created assessment data
        created_assessment = DietaryAssessment.objects.get(user=self.user)
        self.assertEqual(created_assessment.dietary_preferences, self.valid_payload['dietary_preferences'])
        self.assertEqual(created_assessment.activity_levels, self.valid_payload['activity_levels'])
        self.assertEqual(created_assessment.health_goals, self.valid_payload['health_goals'])
        self.assertEqual(created_assessment.goals, self.valid_payload['goals'])
        self.assertIsNotNone(created_assessment.tdee)
        self.assertIsNotNone(created_assessment.bmi)

    def test_update_dietary_assessment(self):
        self.test_assessment = DietaryAssessment.objects.create(
            user=self.user,
            dietary_preferences=[DietaryPreference.VEGETARIAN],
            activity_levels=[ActivityLevel.MODERATELY_ACTIVE],
            health_goals=[HealthGoal.LOSE_WEIGHT],
            goals=['Lose 10 pounds in 3 months']
        )
        self.test_assessment.liked_ingredients.add(self.ingredient1, self.ingredient3)
        self.test_assessment.disliked_ingredients.add(self.ingredient2)
        update_data = {
            'dietary_preferences': ['VEG'],
            'activity_levels': ['VER'],
            'health_goals': ['GAI'],
            'liked_ingredients': ['Broccoli'],
            'disliked_ingredients': ['Tomato'],
            'goals': ['Gain 5 pounds of muscle in 2 months']
        }

        response = self.client.put(self.list_create_url, update_data, format='json')
        # Print response content if there's an error
        if response.status_code != status.HTTP_200_OK:
            print(f"Error response content: {response.content}")
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        # Verify the changes
        updated_assessment = DietaryAssessment.objects.get(user=self.user)
        self.assertEqual(updated_assessment.dietary_preferences, ['VEG'])
        self.assertEqual(updated_assessment.activity_levels, ['VER'])
        self.assertEqual(updated_assessment.health_goals, ['GAI'])
        self.assertEqual(list(updated_assessment.liked_ingredients.values_list('name', flat=True)), ['Broccoli'])
        self.assertEqual(list(updated_assessment.disliked_ingredients.values_list('name', flat=True)), ['Tomato'])
        self.assertEqual(updated_assessment.goals, ['Gain 5 pounds of muscle in 2 months'])



    def test_update_with_invalid_ingredient(self):
        self.test_assessment = DietaryAssessment.objects.create(
            user=self.user,
            dietary_preferences=[DietaryPreference.VEGETARIAN],
            activity_levels=[ActivityLevel.MODERATELY_ACTIVE],
            health_goals=[HealthGoal.LOSE_WEIGHT],
            goals=['Lose 10 pounds in 3 months']
        )
        self.test_assessment.liked_ingredients.add(self.ingredient1, self.ingredient3)
        self.test_assessment.disliked_ingredients.add(self.ingredient2)
        update_data = {
            'liked_ingredients': ['InvalidIngredient'],
        }
        response = self.client.put(self.list_create_url, update_data, format='json')
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    def test_retrieve_dietary_assessment(self):
        # Create a test dietary assessment
        self.test_assessment = DietaryAssessment.objects.create(
            user=self.user,
            dietary_preferences=[DietaryPreference.VEGETARIAN],
            activity_levels=[ActivityLevel.MODERATELY_ACTIVE],
            health_goals=[HealthGoal.LOSE_WEIGHT],
            goals=['Lose 10 pounds in 3 months']
        )
        self.test_assessment.liked_ingredients.add(self.ingredient1, self.ingredient3)
        self.test_assessment.disliked_ingredients.add(self.ingredient2)

        self.retrieve_url = f'http://localhost:8000/dietary-assessment/{self.test_assessment.dietary_assessment_id}/'
        response = self.client.get(self.retrieve_url)
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        
        # Check if the retrieved data matches the test assessment
        self.assertEqual(response.data['user'], self.user.id)
        self.assertEqual(response.data['dietary_preferences'], [DietaryPreference.VEGETARIAN])
        self.assertEqual(response.data['activity_levels'], [ActivityLevel.MODERATELY_ACTIVE])
        self.assertEqual(response.data['health_goals'], [HealthGoal.LOSE_WEIGHT])
        self.assertEqual(response.data['goals'], ['Lose 10 pounds in 3 months'])
        
        # Check liked and disliked ingredients
        liked_ingredients = [ingredient.name for ingredient in self.test_assessment.liked_ingredients.all()]
        disliked_ingredients = [ingredient.name for ingredient in self.test_assessment.disliked_ingredients.all()]
        self.assertIn('Broccoli', liked_ingredients)
        self.assertIn('Tomato', liked_ingredients)
        self.assertIn('Chicken', disliked_ingredients)

    def test_retrieve_nonexistent_assessment(self):
        nonexistent_url = 'https://localhost:8000/dietary-assessment/9999/'
        response = self.client.get(nonexistent_url)
        
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)
        self.assertIn('error', response.data)

    def test_retrieve_other_user_assessment(self):
        # Create another user and assessment
        other_user_data = self.user_data.copy()
        other_user_data['email'] = 'otheruser@example.com'
        other_user_data['username'] = 'OtherUser'
        response = self.client.post(self.register_url, other_user_data, format='json')
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        other_user = get_user_model().objects.get(email=other_user_data['email'])
        
        other_assessment = DietaryAssessment.objects.create(
            user=other_user,
            dietary_preferences=[DietaryPreference.VEGAN],
            activity_levels=ActivityLevel.VERY_ACTIVE,
            health_goals=[HealthGoal.GAIN_WEIGHT],
            goals=['Gain muscle mass']
        )

        # Try to retrieve the other user's assessment
        other_retrieve_url = f'/dietary-assessment/{other_assessment.dietary_assessment_id}/'
        response = self.client.get(other_retrieve_url)
        
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)
        self.assertIn('error', response.data)


class RatingViewTestCase(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.register_url = 'http://localhost:8000/register/'
        self.login_url = 'http://localhost:8000/login/'
        self.rating_url='http://localhost:8000/rating/{}/'
        
        # Create a test recipe
        # Create nutritional information for the recipe
        self.nutritional_info = NutritionalInformation.objects.create(
            calories=200,  
            protein=10,
            fat=5,
            carbs=30,
            fiber=34
        )

        # Valid recipe data
        self.valid_recipe_data = {
            'name': 'Test Recipe 1',
            'ingredients': ['ingredient1', 'ingredient2'],
            'cuisine': ['ghanaian'],  # Ensure the cuisine matches your model's choices
            'recipe_info': 'A test recipe for unit testing',
            'vegan': True,
            'vegetarian': True,
            'gluten_free': False,
            'pescatarian': False,
            'halal': True,
            'meal_type': ['dinner'],  # Ensure meal type matches your model's choices
            'dish_type': ['main'],  # Ensure dish type matches your model's choices
            'tags': ['high-protein'],
            'nutrition': {
                'calories': 200,
                'protein': 10,
                'fat': 5,
                'carbs': 30,
                'fiber': 34
            }
        }

        # Create a recipe for update and delete tests
        nutrition = NutritionalInformation.objects.create(**self.valid_recipe_data['nutrition'])
        recipe_data = self.valid_recipe_data.copy()
        recipe_data['nutrition'] = nutrition
        self.recipe = Recipe.objects.create(**recipe_data)
        
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
    
    def test_create_rating(self):
        data = {'rating': 4,'comment':"Wow!"}
        response = self.client.post(self.rating_url.format(self.recipe.recipe_id), data, format='json')
        if response.status_code != status.HTTP_201_CREATED:
            print(f"Error response content: {response.content}")
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(Rating.objects.count(), 1)
        self.assertEqual(Rating.objects.get().rating, 4)
    
    def test_get_ratings(self):
        # Create a rating first
        Rating.objects.create(user=self.user, recipe=self.recipe, rating=5,comment="Yes")
        
        response = self.client.get(self.rating_url.format(self.recipe.recipe_id))
        if response.status_code != status.HTTP_200_OK:
            print(f"Error response content: {response.content}")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data), 1)
        self.assertEqual(response.data[0]['rating'], 5)

class FavoriteViewTestCase(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.register_url = 'http://localhost:8000/register/'
        self.login_url = 'http://localhost:8000/login/'
        self.favorite_url='http://localhost:8000/favorites/'
        self.favorite_act_url='http://localhost:8000/favorites/{}/'
        
        # Create a test recipe
        # Create a test recipe
        # Create nutritional information for the recipe
        self.nutritional_info = NutritionalInformation.objects.create(
            calories=200,  
            protein=10,
            fat=5,
            carbs=30,
            fiber=34
        )

        # Valid recipe data
        self.valid_recipe_data = {
            'name': 'Test Recipe 1',
            'ingredients': ['ingredient1', 'ingredient2'],
            'cuisine': ['ghanaian'],  # Ensure the cuisine matches your model's choices
            'recipe_info': 'A test recipe for unit testing',
            'vegan': True,
            'vegetarian': True,
            'gluten_free': False,
            'pescatarian': False,
            'halal': True,
            'meal_type': ['dinner'],  # Ensure meal type matches your model's choices
            'dish_type': ['main'],  # Ensure dish type matches your model's choices
            'tags': ['high-protein'],
            'nutrition': {
                'calories': 200,
                'protein': 10,
                'fat': 5,
                'carbs': 30,
                'fiber': 34
            }
        }

        # Create a recipe for update and delete tests
        nutrition = NutritionalInformation.objects.create(**self.valid_recipe_data['nutrition'])
        recipe_data = self.valid_recipe_data.copy()
        recipe_data['nutrition'] = nutrition
        self.recipe = Recipe.objects.create(**recipe_data)
        
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
    
    def test_create_favorite(self):
        response = self.client.post(self.favorite_act_url.format(self.recipe.recipe_id), format='json')
        if response.status_code != status.HTTP_201_CREATED:
            print(f"Error response content: {response.content}")
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(Favorite.objects.count(), 1)
        self.assertEqual(Favorite.objects.get().recipe, self.recipe)
    
    def test_get_favorites(self):
        # Create a favorite first
        Favorite.objects.create(user=self.user, recipe=self.recipe)
        
        response = self.client.get(self.favorite_url.format(self.recipe.recipe_id))
        if response.status_code != status.HTTP_200_OK:
            print(f"Error response content: {response.content}")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data), 1)
        self.assertEqual(response.data[0]['recipe'], self.recipe.recipe_id)
    
    def test_delete_favorite(self):
        # Create a favorite first
        favorite = Favorite.objects.create(user=self.user, recipe=self.recipe)
        
        response = self.client.delete(self.favorite_act_url.format(self.recipe.recipe_id),format='json')
        if response.status_code != status.HTTP_204_NO_CONTENT:
            print(f"Error response content: {response.content}")
        self.assertEqual(response.status_code, status.HTTP_204_NO_CONTENT)
        self.assertEqual(Favorite.objects.count(), 0)
    
    def test_delete_nonexistent_favorite(self):
        response = self.client.delete(self.favorite_act_url.format(self.recipe.recipe_id), format='json')
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

class EducationalContentViewsTestCase(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.list_create_url = 'http://localhost:8000/educational-content/'
        self.content_type_url = 'http://localhost:8000/educational-content-filter/'
        self.specific_url='http://localhost:8000/educational-content/{}/'

        # Create some test data
        self.content1 = EducationalContent.objects.create(
            title="Test Content 1",
            description="Description 1",
            content_type="text",
            content_url="https://www.googel.cone",
            tags=['ds','sdf'],
            content_image=SimpleUploadedFile("file1.png", b"file_content", content_type="media/png")
        )
        self.content2 = EducationalContent.objects.create(
            title="Test Content 2",
            description="Description 2",
            content_type="article",
            content_url="https://www.googel.cone",
            tags=['ds','sdf'],
            content_image=SimpleUploadedFile("file1.png", b"file_content", content_type="media/png")
        )

    def test_list_educational_content(self):
        response = self.client.get(self.list_create_url)
        if response.status_code != status.HTTP_200_OK:
            print(f"Error response content list: {response.content}")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data), 2)

    def test_create_educational_content(self):
        data = {
            'title': 'New Content',
            'description': 'New Description',
            'content_type': 'video',
            'content_url':"https://www.googel.cone",
            'tags':json.dumps(['ds','sdf']),
            'content_image':SimpleUploadedFile("file1.mp4", b"file_content", content_type="application/mp4")
        }
        response = self.client.post(self.list_create_url, data, format='multipart')
        if response.status_code != status.HTTP_201_CREATED:
            print(f"Error response content create: {response.content}")
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(EducationalContent.objects.count(), 3)
        self.assertEqual(response.data['content']['title'], 'New Content')

    def test_retrieve_educational_content(self):
        response = self.client.get(self.specific_url.format(self.content1.content_id))
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['title'], 'Test Content 1')

    def test_update_educational_content(self):
        data = {'title': 'Updated Title'}
        response = self.client.put(self.specific_url.format(self.content1.content_id),data, format='multipart')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['content']['title'], 'Updated Title')

    def test_delete_educational_content(self):
        response = self.client.delete(self.specific_url.format(self.content1.content_id))
        if response.status_code != status.HTTP_204_NO_CONTENT:
            print(f"Error response content delete: {response.content}")
        self.assertEqual(response.status_code, status.HTTP_204_NO_CONTENT)
        self.assertEqual(EducationalContent.objects.count(), 1)

    def test_list_educational_content_by_type(self):
        response = self.client.get(f"{self.content_type_url}?type=text")
        if response.status_code != status.HTTP_200_OK:
            print(f"Error response content typw: {response.content}")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data), 1)
        self.assertEqual(response.data[0]['title'], 'Test Content 1')

    def test_list_educational_content_by_type_no_param(self):
        response = self.client.get(self.content_type_url)
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    def test_retrieve_nonexistent_content(self):
        response = self.client.get(self.specific_url.format('2ftgregeer'))
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

    def test_update_nonexistent_content(self):
        data = {'title': 'Updated Title'}
        response = self.client.put(self.specific_url.format('2ftgregeer'), data, format='multipart')
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

    def test_delete_nonexistent_content(self):
        response = self.client.delete(self.specific_url.format('2ftgregeer'))
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)
