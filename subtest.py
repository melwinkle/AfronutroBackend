
class DietaryAssessmentAPITestCase(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create_user(email='testuser@example.com', password='testpass123',weight=74.0, height=157.0,age=24,gender="Male")
        self.client.force_authenticate(user=self.user)

    def test_create_dietary_assessment(self):
        url = reverse('dietary_assessment')
        data = {
            'dietary_preferences': ['VEG', 'GLU'],
            'activity_levels': ['MOD'],
            'health_goals': ['LOS', 'FIT'],
            'liked_ingredients': ['tomato', 'spinach'],
            'disliked_ingredients': ['onion'],
            'goals': ['Lose weight','improve fitness']
        }
        response = self.client.post(url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertTrue(DietaryAssessment.objects.filter(user=self.user).exists())

    def test_update_dietary_assessment(self):
        assessment = DietaryAssessment.objects.create(
            user=self.user,
            dietary_preferences=['VEG'],
            activity_levels=['LIG'],
            health_goals=['MAI'],
            goals=['Be strong']
        )
        url = reverse('dietary_assessment')
        updated_data = {
            'dietary_preferences': ['VEG', 'GLU'],
            'activity_levels': ['MOD'],
            'health_goals': ['LOS', 'FIT']
        }
        response = self.client.put(url, updated_data, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        assessment.refresh_from_db()
        self.assertEqual(assessment.dietary_preferences, ['VEG', 'GLU'])
        self.assertEqual(assessment.activity_levels, ['MOD'])
        self.assertEqual(assessment.health_goals, ['LOS', 'FIT'])

class MealPlanAPITestCase(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create_user(email='testuser@example.com', password='testpass123',weight=74.0, height=157.0,age=24,gender="Male")
        self.client.force_authenticate(user=self.user)
        self.recipe = Recipe.objects.create(
            name='Tofu Light Soup',
            ingredients=["tofu","tomatoes","onions","pepper","garden eggs","garlic", "ginger", "onions", "kpakposhito paste","dry spices","salt", "stock cube"],
            recipe_info='Low fat, Low calorie',
            cuisine=['ghanain'],
            vegan=True,
            vegetarian=True,
            gluten_free=True,
            pescatarian=False,
            halal=True,
            meal_type=['lunch','dinner'],
            dish_type=['soup', 'one-pot'],
            tags=['low-fat','low-calorie'],
            nutrition={
                "calories":262,
                "protein": 14,
                "fat":7,
                "carbs":39,
                "fiber":8
            }
            
        )
        self.dietary_assessment = DietaryAssessment.objects.create(
            user=self.user,
            dietary_preferences=['VEG'],
            activity_levels=['MOD'],
            health_goals=['LOS']
        )

    @patch('users.views.cache.get')
    def test_generate_meal_plan(self, mock_cache_get):
        mock_recommender = mock_cache_get.return_value
        mock_recommender.get_recommendations.return_value = {
            'breakfast': ['Test Recipe'],
            'lunch': ['Test Recipe'],
            'dinner': ['Test Recipe']
        }
        
        url = reverse('generate_meal_plan')
        response = self.client.post(url)
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertTrue(MealPlan.objects.filter(user=self.user).exists())

    def test_customize_meal_plan(self):
        meal_plan = MealPlan.objects.create(user=self.user, name='Test Meal Plan')
        url = reverse('customize_meal_plan', kwargs={'meal_plan_id': meal_plan.meal_plan_id})
        data = {
            'name': 'Updated Meal Plan',
            'description': 'Updated description'
        }
        response = self.client.put(url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        meal_plan.refresh_from_db()
        self.assertEqual(meal_plan.name, 'Updated Meal Plan')
        self.assertEqual(meal_plan.description, 'Updated description')

    def test_delete_meal_plan(self):
        meal_plan = MealPlan.objects.create(user=self.user, name='Test Meal Plan')
        url = reverse('get_meal_plan_by_id', kwargs={'meal_plan_id': meal_plan.meal_plan_id})
        response = self.client.delete(url)
        self.assertEqual(response.status_code, status.HTTP_204_NO_CONTENT)
        self.assertFalse(MealPlan.objects.filter(id=meal_plan.id).exists())

class IngredientAPITestCase(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create_user(email='testuser@example.com', password='testpass123',weight=74.0, height=157.0,age=24,gender="Male")
        self.client.force_authenticate(user=self.user)
        self.ingredient = Ingredient.objects.create(name='Test Ingredient',calories=234,protein=23,fat=2,carbs=2,vitamins=["b2"],minerals=["Phosporus"],substitutes=[])

    def test_create_ingredient(self):
        url = reverse('ingredients')
        data = {
            "name": "Garden eggs",
            "calories": 36,
            "protein": 1.5,
            "fat": 0.5,
            "carbs": 8,
            "vitamins": ["Vitamin B1"],
            "minerals": ["Potassium", "Fiber"],
            "substitutes": []
        }
        response = self.client.post(url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertTrue(Ingredient.objects.filter(name='Garden eggs').exists())

    def test_get_ingredients(self):
        url = reverse('ingredient')
        response = self.client.get(url)
        ingredients = Ingredient.objects.all()
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data), ingredients.count())

class RatingAPITestCase(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create_user(email='testuser@example.com', password='testpass123',weight=74.0, height=157.0,age=24,gender="Male")
        self.client.force_authenticate(user=self.user)
        self.recipe = Recipe.objects.create(
            name='Test Recipe',
            ingredients=["tofu","tomatoes","onions","pepper","garden eggs","garlic", "ginger", "onions", "kpakposhito paste","dry spices","salt", "stock cube"],
            recipe_info='Low fat, Low calorie',
            cuisine=['ghanain'],
            vegan=True,
            vegetarian=True,
            gluten_free=True,
            pescatarian=False,
            halal=True,
            meal_type=['lunch','dinner'],
            dish_type=['soup', 'one-pot'],
            tags=['low-fat','low-calorie'],
            nutrition={
                "calories":262,
                "protein": 14,
                "fat":7,
                "carbs":39,
                "fiber":8
            }
            )

    def test_create_rating(self):
        url = reverse('rating-view', kwargs={'recipe_id': self.recipe.recipe_id})
        data = {'rating': 4, 'review': 'Great recipe!'}
        response = self.client.post(url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertTrue(Rating.objects.filter(user=self.user, recipe=self.recipe).exists())

class FavoriteAPITestCase(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create_user(email='testuser@example.com', password='testpass123',weight=74.0, height=157.0,age=24,gender="Male")
        self.client.force_authenticate(user=self.user)
        self.recipe = Recipe.objects.create(
            name='Test Recipe',
            ingredients=["tofu","tomatoes","onions","pepper","garden eggs","garlic", "ginger", "onions", "kpakposhito paste","dry spices","salt", "stock cube"],
            recipe_info='Low fat, Low calorie',
            cuisine=['ghanain'],
            vegan=True,
            vegetarian=True,
            gluten_free=True,
            pescatarian=False,
            halal=True,
            meal_type=['lunch','dinner'],
            dish_type=['soup', 'one-pot'],
            tags=['low-fat','low-calorie'],
            nutrition={
                "calories":262,
                "protein": 14,
                "fat":7,
                "carbs":39,
                "fiber":8
            })

    def test_add_favorite(self):
        url = reverse('favorite-view', kwargs={'recipe_id': self.recipe.recipe_id})
        response = self.client.post(url)
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertTrue(Favorite.objects.filter(user=self.user, recipe=self.recipe).exists())

    def test_remove_favorite(self):
        Favorite.objects.create(user=self.user, recipe=self.recipe)
        url = reverse('favorite-view', kwargs={'recipe_id': self.recipe.recipe_id})
        response = self.client.delete(url)
        self.assertEqual(response.status_code, status.HTTP_204_NO_CONTENT)
        self.assertFalse(Favorite.objects.filter(user=self.user, recipe=self.recipe).exists())

class EducationalContentAPITestCase(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create_user(email='testuser@example.com', password='testpass123',weight=74.0, height=157.0,age=24,gender="Male")
        self.client.force_authenticate(user=self.user)
        self.content = EducationalContent.objects.create(
            title='Test Content',
            content_type='ART',
            description='Test educational content',
            content_url='https://www.google.com',
            tags=['cont']
        )

    def test_create_educational_content(self):
        url = reverse('educational-content-list-create')
        data = {
            'title': 'New Content',
            'content_type': 'video',
            'description': 'New educational content',
            'content_url':'https://www.google.com',
           ' tags':['cont']
            
        }
        response = self.client.post(url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertTrue(EducationalContent.objects.filter(title='New Content').exists())

    def test_get_educational_content(self):
        url = reverse('educational-content-detail', kwargs={'content_id': self.content.content_id})
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['title'], 'Test Content')

    def test_update_educational_content(self):
        url = reverse('educational-content-detail', kwargs={'content_id': self.content.content_id})
        data = {'title': 'Updated Content'}
        response = self.client.put(url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.content.refresh_from_db()
        self.assertEqual(self.content.title, 'Updated Content')

    def test_delete_educational_content(self):
        url = reverse('educational-content-detail', kwargs={'content_id': self.content.content_id})
        response = self.client.delete(url)
        self.assertEqual(response.status_code, status.HTTP_204_NO_CONTENT)
        self.assertFalse(EducationalContent.objects.filter(id=self.content.id).exists())

# Add more test cases as needed