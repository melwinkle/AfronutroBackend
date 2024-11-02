from django.test import TestCase
from .models import Recipe, Rating, Favorite, NutritionalInformation,Ingredient
from .serializers import NutritionalInformationSerializer,RecipeSerializer,FavoriteSerializer,RatingSerializer,RecipeListSerializer,RecipeFilterSerializer,RecipeSearchSerializer

# Create your tests here.
User = get_user_model()
print(f"Test database name: {connection.settings_dict['NAME']}")
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

