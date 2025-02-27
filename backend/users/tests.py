from django.test import TestCase
from django.urls import reverse
from rest_framework.test import APIClient
from rest_framework import status
from rest_framework.authtoken.models import Token
from django.contrib.auth import get_user_model
from .models import  DietaryAssessment, EducationalContent, DietaryPreference,ActivityLevel,HealthGoal
from recipes.models import CuisineType
from .serializers import  DietaryAssessmentSerializer
from recipes.models import Ingredient
from django.core.cache import cache
from unittest.mock import patch
from django.db import connection
import json
from django.core.files.uploadedfile import SimpleUploadedFile
from datetime import datetime


User = get_user_model()
print(f"Test database name: {connection.settings_dict['NAME']}")

class UserAPITestCase(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.register_url = 'http://localhost:8000/register/'  # Adjust URL for Node.js
        self.login_url = 'http://localhost:8000/login/'
        self.logout_url = 'http://localhost:8000/logout/'

        self.user_data={
            "email": "testusers@example.com",
            "username": "User",
            "password": "password123$",
            "password2": "password123$",
            "date_of_birth": '2000-12-31',
            "activity_levels":1.375,
            "tdee":1882, 
            "bmi":27.34,
            "gender": "Female",
            "height": 160.0,
            "weight": 70,
            "is_verified": False,  
        }
        

    def test_user_registration(self):

        response = self.client.post(self.register_url, self.user_data, format='json')
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        created_user = get_user_model().objects.get(email=self.user_data['email'])
        self.assertEqual(created_user.username, self.user_data['username'])
        self.assertEqual(created_user.date_of_birth, datetime.strptime(self.user_data['date_of_birth'], '%Y-%m-%d').date())
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
        self.assertIn('auth_token', login_response.cookies)
        self.assertIn('user', login_response.data)
        self.assertIn('user_id', login_response.data['user'])
        self.assertIn('email', login_response.data['user'])

        # Store the token for further testing (e.g., logout)
        self.token = login_response.cookies['auth_token'].value 

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
        self.client.logout()

        # Attempt to register with the same email
        invalid_register_data = {
            'email': self.user_data['email'],
            'username': self.user_data['username'],
            'password': 'newpassword$',
            'password2': "newpassword$",
            'date_of_birth': self.user_data['date_of_birth'],
            'gender': 'Male',
            'height': 157.0,
            'weight': 68,
            'tdee':1800,
            'bmi':27.32,
            'activity_levels':1.375,
        }
        invalid_register_response = self.client.post(self.register_url, invalid_register_data, format='json')

        # Assert registration with the same email fails
        if invalid_register_response.status_code != status.HTTP_400_BAD_REQUEST:
           print("Registration failed:", invalid_register_response.data)
        # Check if the error messages match what you expect
        if 'email' in invalid_register_response.data:
            self.assertEqual(invalid_register_response.data['email'][0].code, 'unique')
        else:
            print("Response data:", invalid_register_response.data)  # Print the response for debugging







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
            "date_of_birth": '2000-12-31',
            "activity_levels":1.375,
            "tdee":1882, 
            "bmi":27.34,
            "gender": "Male",
            "height": 160.0,
            "weight": 70,
            "is_verified": True,
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
        self.assertIn('auth_token', login_response.cookies)
 

        # Store the token and set up authentication for further requests
        self.token = login_response.cookies['auth_token'].value 
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
            'cuisine_preference':['ghanaian']
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
        self.assertEqual(created_assessment.cuisine_preference, self.valid_payload['cuisine_preference'])
        self.assertIsNotNone(created_assessment.tdee)
        self.assertIsNotNone(created_assessment.bmi)

    def test_update_dietary_assessment(self):
        self.test_assessment = DietaryAssessment.objects.create(
            user=self.user,
            dietary_preferences=[DietaryPreference.VEGETARIAN],
            activity_levels=[ActivityLevel.MODERATELY_ACTIVE],
            health_goals=[HealthGoal.LOSE_WEIGHT],
        )
        self.test_assessment.liked_ingredients.add(self.ingredient1, self.ingredient3)
        self.test_assessment.disliked_ingredients.add(self.ingredient2)
        update_data = {
            'dietary_preferences': ['VEG'],
            'activity_levels': ['VER'],
            'health_goals': ['GAI'],
            'liked_ingredients': ['Broccoli'],
            'disliked_ingredients': ['Tomato'],
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



    def test_update_with_invalid_ingredient(self):
        self.test_assessment = DietaryAssessment.objects.create(
            user=self.user,
            dietary_preferences=[DietaryPreference.VEGETARIAN],
            activity_levels=[ActivityLevel.MODERATELY_ACTIVE],
            health_goals=[HealthGoal.LOSE_WEIGHT],
        )
        self.test_assessment.liked_ingredients.add(self.ingredient1, self.ingredient3)
        self.test_assessment.disliked_ingredients.add(self.ingredient2)
        update_data = {
            'liked_ingredients': ['InvalidIngredient'],
        }
        response = self.client.put(self.list_create_url, update_data, format='json')
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    # def test_retrieve_dietary_assessment(self):
    #     # Create a test dietary assessment
    #     self.test_assessment = DietaryAssessment.objects.create(
    #         user=self.user,
    #         dietary_preferences=[DietaryPreference.VEGETARIAN],
    #         activity_levels=[ActivityLevel.MODERATELY_ACTIVE],
    #         health_goals=[HealthGoal.LOSE_WEIGHT],
    #         cuisine_preference=[CuisineType.GHANAIAN],
    #     )
    #     self.test_assessment.liked_ingredients.add(self.ingredient1, self.ingredient3)
    #     self.test_assessment.disliked_ingredients.add(self.ingredient2)


    #     self.retrieve_url = f'http://localhost:8000/dietary-assessment/{self.test_assessment.dietary_assessment_id}/'
    #     response = self.client.get(self.retrieve_url)
        

    #     if response.status_code != status.HTTP_200_OK:
    #         print(f"Response Failed: {response}")
        
    #     self.assertEqual(response.status_code, status.HTTP_200_OK)
        
 

        
    #     # Check liked and disliked ingredients
    #     liked_ingredients = [ingredient.name for ingredient in self.test_assessment.liked_ingredients.all()]
    #     disliked_ingredients = [ingredient.name for ingredient in self.test_assessment.disliked_ingredients.all()]
    #     self.assertIn('Broccoli', liked_ingredients)
    #     self.assertIn('Tomato', liked_ingredients)
    #     self.assertIn('Chicken', disliked_ingredients)
    
    def test_retrieve_dietary_assessment(self):
        # Create a dietary assessment
        assessment_data = {
            'dietary_preferences': ['VGT'],
            'activity_levels': ['MOD'],
            'health_goals': ['LOS'],
            'liked_ingredients': ['Broccoli', 'Tomato'],
            'disliked_ingredients': ['Chicken'],
            'cuisine_preference': ['ghanaian'],
        }
        responses=self.client.post(self.list_create_url, assessment_data, format='json')
        id = responses.data['data']['dietary_assessment_id']
        if responses.status_code != status.HTTP_201_CREATED:
            print(f"Error response content: {responses.content}")
        

        self.retrieve_url = f'http://localhost:8000/dietary-assessment-view/'
        response = self.client.get(self.retrieve_url)
        
     

        

    def test_retrieve_nonexistent_assessment(self):
        nonexistent_url = 'https://localhost:8000/dietary-assessment/9999/'
        response = self.client.get(nonexistent_url)
        
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)
        if hasattr(response, 'data'):
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
            cuisine_preference=[CuisineType.CHINESE]
        )

        # Try to retrieve the other user's assessment
        other_retrieve_url = f'/dietary-assessment/{other_assessment.dietary_assessment_id}/'
        response = self.client.get(other_retrieve_url)
        
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)
        if hasattr(response, 'data'):
            self.assertIn('error', response.data)



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
