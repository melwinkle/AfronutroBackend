from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import generics, permissions, status, viewsets
from rest_framework.authtoken.views import ObtainAuthToken,APIView
from django.core.cache import cache
from django.utils import timezone
from .models import MealPlan
from .serializers import MealPlanSerializer
from users.models import DietaryAssessment, DietaryPreference
from recipes.models import Recipe
from .tasks import fit_recommender_task
from celery.result import AsyncResult
from typing import Dict, Any, Union, Optional,List
import logging
import json
from django.core.exceptions import ValidationError

logger = logging.getLogger(__name__)


class GenerateMealPlanView(APIView):
    def post(self, request) -> Response:
        """
        Generate a meal plan for the user.
        
        Returns:
            Response: API response with meal plan or error details
        """
        try:
            # Validate user authentication
            if not request.user.is_authenticated:
                return Response(
                    {"error": "Authentication required"},
                    status=status.HTTP_401_UNAUTHORIZED
                )

            # Add debug logging for user profile preparation
            logger.debug("Starting user profile preparation")
            try:
                user_profile = self._prepare_user_profile(request)
                logger.debug(f"Generated user profile: {json.dumps(user_profile, default=str)}")
            except ValueError as e:
                logger.error(f"User profile preparation failed: {str(e)}")
                return Response(
                    {"error": str(e)},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Get recommender with debug logging
            logger.debug("Getting recommender")
            recommender = self._get_or_train_recommender()
            if isinstance(recommender, Response):
                return recommender
                
            # Generate recommendations with debug logging
            logger.debug("Generating recommendations")
            try:
                meal_plans = recommender.get_recommendations(user_profile)
                logger.debug(f"Generated meal plans: {meal_plans}")
            except Exception as e:
                logger.error(f"Recommendation generation failed: {str(e)}", exc_info=True)
                return Response(
                    {"error": "Failed to generate recommendations", "detail": str(e)},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )

            # Step 4: Create meal plan with validation
            try:
                meal_plan = self._create_meal_plan(request.user, meal_plans)
            except Recipe.DoesNotExist as e:
                logger.error(f"Recipe not found during meal plan creation: {str(e)}")
                return Response(
                    {"error": "Some recipes are no longer available"},
                    status=status.HTTP_404_NOT_FOUND
                )
            except ValidationError as e:
                logger.error(f"Meal plan validation failed: {str(e)}")
                return Response(
                    {"error": str(e)},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Step 5: Serialize and return response
            try:
                serializer = MealPlanSerializer(meal_plan)
                return Response(serializer.data, status=status.HTTP_201_CREATED)
            except ValidationError as e:
                logger.error(f"Serialization failed: {str(e)}")
                return Response(
                    {"error": "Failed to serialize meal plan"}, 
                    status=status.HTTP_400_BAD_REQUEST
    )
            except Exception as e:
                logger.error(f"Serialization failed: {str(e)}")
                return Response(
                    {"error": "Failed to process meal plan data"},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )

        except Exception as e:
            logger.error(f"Unexpected error in meal plan generation: {str(e)}", exc_info=True)
            return Response(
                {"error": "An unexpected error occurred"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    def _prepare_user_profile(self, request) -> Dict[str, Any]:
        """
        Prepare user profile for recommendation with validation.
        
        Raises:
            ValueError: If required user data is missing
            DietaryAssessment.DoesNotExist: If assessment not found
        """
        # Validate user data
        required_fields = ['age', 'gender', 'weight', 'height']
        missing_fields = [field for field in required_fields 
                        if not getattr(request.user, field, None)]
        if missing_fields:
            raise ValueError(f"Missing required user data: {', '.join(missing_fields)}")

        try:
            assessment = request.user.dietaryassessment
        except DietaryAssessment.DoesNotExist:
            raise

        # Validate assessment data
        required_assessment_fields = ['activity_levels', 'tdee', 'bmi', 'health_goals']
        missing_assessment = [field for field in required_assessment_fields 
                            if not getattr(assessment, field, None)]
        if missing_assessment:
            raise ValueError(f"Incomplete dietary assessment: {', '.join(missing_assessment)}")

        print("Assessment dietary preferences:", assessment.dietary_preferences)
        print("Type of dietary preferences:", type(assessment.dietary_preferences))
        
        dietary_codes = self._get_dietary_codes(assessment)
        if not self._validate_dietary_codes(dietary_codes):
            logger.warning(f"Invalid dietary codes generated: {dietary_codes}")
            dietary_codes = []
        print("Generated codes:", dietary_codes)

        # Convert Ingredient objects to strings (names)
        liked_ingredients = [ingredient.name for ingredient in assessment.liked_ingredients.all()]
        disliked_ingredients = [ingredient.name for ingredient in assessment.disliked_ingredients.all()]
        

        return {
            'user_id': request.user.id,
            'age': request.user.age,
            'gender': request.user.gender,
            'weight': request.user.weight,
            'height': request.user.height,
            'activity_level': assessment.activity_levels,
            'tdee': assessment.tdee,
            'bmi': assessment.bmi,
            'liked_ingredients': liked_ingredients,  # Now using string names instead of objects
            'disliked_ingredients': disliked_ingredients,  # Now using string names instead of objects
            'dietary_preferences': dietary_codes,
            'health_goals': assessment.health_goals,
            'cuisine_preference':assessment.cuisine_preference
        }

    # Also add this validation method to your GenerateMealPlanView class
    def _validate_user_profile(self, profile: Dict[str, Any]) -> bool:
        """
        Validate that all profile values are of the correct type
        """
        try:
            # Check that ingredients are lists of strings
            if not all(isinstance(x, str) for x in profile['liked_ingredients']):
                logger.error("Liked ingredients contains non-string values")
                return False
            if not all(isinstance(x, str) for x in profile['disliked_ingredients']):
                logger.error("Disliked ingredients contains non-string values")
                return False
                
            # Validate other critical fields
            if not isinstance(profile['user_id'], int):
                logger.error("user_id is not an integer")
                return False
                
            numeric_fields = ['age', 'weight', 'height', 'tdee', 'bmi']
            for field in numeric_fields:
                if not isinstance(profile[field], (int, float)):
                    logger.error(f"{field} is not a number")
                    return False
                    
            return True
        except KeyError as e:
            logger.error(f"Missing required field in profile: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error validating profile: {e}")
            return False

    def _get_dietary_codes(self, assessment: DietaryAssessment) -> list:
        """
        Get dietary preference codes with validation.
        
        Args:
            assessment: DietaryAssessment instance
            
        Returns:
            list: List of dietary preference codes
        """
        try:
            logger.debug("Starting dietary code generation")
            
            # Initialize empty codes list
            codes = []
            
            # First handle the dietary_preferences list that's already in code format
            if hasattr(assessment, 'dietary_preferences') and assessment.dietary_preferences:
                if isinstance(assessment.dietary_preferences, list):
                    codes.extend(assessment.dietary_preferences)
                elif isinstance(assessment.dietary_preferences, str):
                    try:
                        parsed_prefs = json.loads(assessment.dietary_preferences)
                        if isinstance(parsed_prefs, list):
                            codes.extend(parsed_prefs)
                    except json.JSONDecodeError:
                        # If it's a single string code, add it
                        codes.append(assessment.dietary_preferences)
            
            logger.debug(f"Codes after dietary_preferences: {codes}")
            
            # Now handle boolean fields separately, but don't duplicate codes
            boolean_preferences = {
                'vegetarian': 'VER',
                'vegan': 'VEG',
                'gluten_free': 'GLU',
                'halal': 'HAL',
                'pescatarian': 'PES'
            }
            
            # Check boolean fields and add their codes if True
            for field, code in boolean_preferences.items():
                value = getattr(assessment, field, False)
                logger.debug(f"Checking {field}: {value}")
                if value is True and code not in codes:  # Only add if not already present
                    codes.append(code)
            
            logger.debug(f"Final dietary codes: {codes}")
            return codes
            
        except Exception as e:
            logger.error(f"Error in _get_dietary_codes: {str(e)}", exc_info=True)
            return []

    # Add this validation method to your view class
    def _validate_dietary_codes(self, codes: list) -> bool:
        """
        Validate dietary codes format and values
        """
        valid_codes = {'VER', 'VEG', 'GLU', 'HAL', 'PES'}
        try:
            return (
                isinstance(codes, list) and
                all(isinstance(code, str) for code in codes) and
                all(code in valid_codes for code in codes)
            )
        except Exception as e:
            logger.error(f"Error validating dietary codes: {str(e)}")
            return False

    def _get_or_train_recommender(self) -> Union[Response, Any]:
        """Get cached recommender or trigger training with validation."""
        recommender = cache.get('hybrid_recommender')
        if not recommender:
            try:
                task = fit_recommender_task.delay()
                return Response({
                    "message": "Building recommendations...",
                    "task_id": task.id
                }, status=status.HTTP_202_ACCEPTED)
            except Exception as e:
                logger.error(f"Failed to start recommender training task: {str(e)}")
                return Response({
                    "error": "Failed to initialize recommendation system"
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        return recommender

    def _validate_meal_plans(self, meal_plans: Dict) -> bool:
        """
        Validate the structure and content of generated meal plans.
        """
        if not isinstance(meal_plans, dict):
            return False
            
        required_meals = {'breakfast', 'lunch', 'dinner', 'snack'}
        if not all(meal in meal_plans for meal in required_meals):
            return False
            
        for meal_type, recipes in meal_plans.items():
            if not isinstance(recipes, list) or not recipes:
                return False
                
        return True

    def _create_meal_plan(self, user, recommendations: Dict[str, list]) -> MealPlan:
        """
        Create meal plan from recommendations with validation.
        
        Returns:
            A dictionary of meal types with recipes as lists.
        
        Raises:
            ValidationError: If meal plan creation fails.
            Recipe.DoesNotExist: If recipes not found.
        """
        try:
            # Create a draft meal plan
            # Create tags from dietary assessment
            dietary_assessment = DietaryAssessment.objects.get(user=user)
            tags = {
                "health_goals": dietary_assessment.health_goals,
                "dietary_preferences": dietary_assessment.dietary_preferences,
                "cuisine_preference":dietary_assessment.cuisine_preference
            }
            
            # Create a dictionary to store meals separated by meal type
            meals_structure = {
                'breakfast': [],
                'lunch': [],
                'dinner': [],
                'snack': []
            }

            meal_plan = MealPlan.objects.create(
                user=user,
                name=self._generate_meal_plan_name(dietary_assessment.dietary_preferences),
                tags=tags,
                meals_structure=meals_structure,
                description="Automatically generated based on your dietary assessment.",
                status=MealPlan.DRAFT
            )
        except Exception as e:
            raise ValidationError(f"Failed to create meal plan: {str(e)}")

        added_recipes = 0
        used_recipes = set()  # Keep track of recipes already assigned to a meal type


        # Iterate through each meal type and add recipes
        for meal_type, recipes in recommendations.items():
            for recipe_name in recipes:
                # Skip the recipe if it has already been assigned to a different meal type
                if recipe_name in used_recipes:
                    continue

                try:
                    recipe = Recipe.objects.get(name=recipe_name)
                    meal_plan.meals.add(recipe)

                    # Only add the recipe if it hasn't been used yet
                    if recipe_name not in used_recipes:
                        # Add recipe to the correct meal type in the dictionary
                        meal_plan.meals_structure[meal_type].append(recipe.name)
                        added_recipes += 1
                        used_recipes.add(recipe_name)  # Mark recipe as used for this meal plan

                except Recipe.DoesNotExist:
                    logger.warning(f"Recipe not found: {recipe_name}")
                    continue

        if added_recipes == 0:
            # If no valid recipes were added, delete the empty meal plan
            meal_plan.delete()
            raise ValidationError("No valid recipes found for meal plan")

        # Save the updated meals_structure
        meal_plan.save()
         # Clear any existing draft caches for this user before setting new one
        self._clear_user_draft_caches(user)
    
        # Cache the draft meal plan
        cache_key = f'meal_plan_draft_{meal_plan.meal_plan_id}'
        try:
            cache.set(cache_key, meal_plan, 60*60*24)
        except Exception as e:
            logger.warning(f"Failed to cache meal plan: {str(e)}")

        # Return the structured meals by type
        return meal_plan

    def _generate_meal_plan_name(self, dietary_preferences: List[str]) -> str:
        """
        Generate a meal plan name using all dietary preferences.
        """
        if dietary_preferences:
            # Map all preference codes to their full names
            preference_names = [dict(DietaryPreference.choices).get(pref_code, "") for pref_code in dietary_preferences]
            # Filter out any empty strings and join them with spaces
            meal_plan_name = " ".join(filter(None, preference_names)) + " Meal Plan"
        else:
            # Default name if no preference is available
            meal_plan_name = "Custom Meal Plan"

        return meal_plan_name

    
    def get(self, request, meal_plan_id=None):
        # Case 1: Get specific meal plan by ID
        if meal_plan_id:
            try:
                # First try to get from database
                meal_plan = MealPlan.objects.get(meal_plan_id=meal_plan_id, user=request.user)
                serializer = MealPlanSerializer(meal_plan)
                return Response(serializer.data, status=status.HTTP_200_OK)
            except MealPlan.DoesNotExist:
                # If not in database, check cache
                cached_plan = cache.get(f'meal_plan_draft_{meal_plan_id}')
                if cached_plan and cached_plan.user == request.user:
                    serializer = MealPlanSerializer(cached_plan)
                    return Response(serializer.data, status=status.HTTP_200_OK)
                
                # Not found in either database or cache
                return Response(
                    {"error": "Meal plan not found."}, 
                    status=status.HTTP_404_NOT_FOUND
                )

        # Case 2: Get all meal plans
        # First get all database meal plans
        meal_plans = list(MealPlan.objects.filter(user=request.user))
        db_meal_plan_ids = {plan.meal_plan_id for plan in meal_plans}
        
        # Get draft plans from cache that don't exist in database
        cache_keys = cache.keys('meal_plan_draft_*')
        for key in cache_keys:
            draft_plan = cache.get(key)
            if draft_plan and draft_plan.user == request.user:
                # Only add cached plan if it's not already in database
                if draft_plan.meal_plan_id not in db_meal_plan_ids:
                    meal_plans.append(draft_plan)
                else:
                    # Clear cache if plan exists in database
                    cache.delete(key)

        serializer = MealPlanSerializer(meal_plans, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)
    
    def delete(self, request, meal_plan_id):
        try:
            meal_plan = MealPlan.objects.get(user=request.user, meal_plan_id=meal_plan_id)
            
            # Delete from cache first
            cache_key = f'meal_plan_draft_{meal_plan_id}'
            cache.delete(cache_key)
            
            # Then delete from database
            meal_plan.delete()
            return Response(status=status.HTTP_204_NO_CONTENT)
        except MealPlan.DoesNotExist:
            # Still try to delete from cache even if not in database
            cache_key = f'meal_plan_draft_{meal_plan_id}'
            cache.delete(cache_key)
            return Response({'error': 'Meal plan not found'}, status=status.HTTP_404_NOT_FOUND)

    def _clear_user_draft_caches(self, user):
        """
        Clear all draft meal plan caches for a specific user.
        """
        cache_keys = cache.keys('meal_plan_draft_*')
        for key in cache_keys:
            draft_plan = cache.get(key)
            if draft_plan and draft_plan.user == user:
                cache.delete(key)
class SaveMealPlanView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request, meal_plan_id):
        # Get the draft from cache
        meal_plan = cache.get(f'meal_plan_draft_{meal_plan_id}')
        if not meal_plan:
            return Response({"error": "Draft meal plan not found."}, status=status.HTTP_404_NOT_FOUND)
        
        try:
            # Ensure meals_structure is saved properly
            meal_plan.status = MealPlan.SAVED
            meal_plan.save()  # This will trigger the custom save method
            
            # Remove from cache
            cache.delete(f'meal_plan_draft_{meal_plan_id}')
            
            serializer = MealPlanSerializer(meal_plan)
            return Response(serializer.data, status=status.HTTP_200_OK)
            
        except Exception as e:
            return Response(
                {"error": f"Failed to save meal plan: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class CustomizeMealPlanView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def put(self, request, meal_plan_id):
        meal_plan = cache.get(f'meal_plan_draft_{meal_plan_id}')
        if not meal_plan:
            try:
                meal_plan = MealPlan.objects.get(meal_plan_id=meal_plan_id, user=request.user)
            except MealPlan.DoesNotExist:
                return Response({"error": "Meal plan not found."}, status=status.HTTP_404_NOT_FOUND)

        serializer = MealPlanSerializer(meal_plan, data=request.data, partial=True)
        if serializer.is_valid():
            meals_structure = request.data.get('meals_structure', {})
            for meal_type, recipes in meals_structure.items():
                meal_plan.meals_structure[meal_type] = []
                for recipe_name in recipes:
                    try:
                        recipe = Recipe.objects.get(name=recipe_name)
                        meal_plan.meals.add(recipe)  # This adds the entire recipe object
                        meal_plan.meals_structure[meal_type].append(recipe.name)  # Store just the name
                    except Recipe.DoesNotExist:
                        logger.warning(f"Recipe not found: {recipe_name}")
                        continue

            updated_meal_plan = serializer.save()

            # If it's a draft, update the cache
            if updated_meal_plan.status == MealPlan.DRAFT:
                cache.set(f'meal_plan_draft_{meal_plan_id}', updated_meal_plan, 60*60*24)

            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    
class NutritionalSummaryView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request, meal_plan_id):
        try:
            meal_plan = MealPlan.objects.get(meal_plan_id=meal_plan_id, user=request.user)
        except MealPlan.DoesNotExist:
            return Response({"error": "Meal plan not found."}, status=status.HTTP_404_NOT_FOUND)

        try:
            dietary_assessment = DietaryAssessment.objects.get(user=request.user)
        except DietaryAssessment.DoesNotExist:
            return Response({"error": "Dietary assessment not found."}, status=status.HTTP_404_NOT_FOUND)

        tdee = dietary_assessment.tdee

        # Use the modified function to get the nutritional summary by meal type
        nutritional_summary = meal_plan.calculate_nutritional_composition()

        # Calculate the total nutritional values from the breakdown
        total_nutritional = {
            "calories": sum([nutritional_summary[meal]["calories"] for meal in nutritional_summary]),
            "protein": sum([nutritional_summary[meal]["protein"] for meal in nutritional_summary]),
            "carbs": sum([nutritional_summary[meal]["carbs"] for meal in nutritional_summary]),
            "fat": sum([nutritional_summary[meal]["fat"] for meal in nutritional_summary]),
        }

        # Add TDEE and calorie difference to the total summary
        summary = {
            "nutritional_by_meal_type": nutritional_summary,  # Breakdown by meal type
            "total_nutritional": total_nutritional,           # Totals
            "tdee": tdee,                                     # User's TDEE
            "calorie_difference": total_nutritional["calories"] - tdee  # Calorie difference from TDEE
        }

        return Response(summary)



# class SetTagsView(APIView):
#     permission_classes = [permissions.IsAuthenticated]

#     def post(self, request, meal_plan_id):
#         try:
#             meal_plan = MealPlan.objects.get(meal_plan_id=meal_plan_id, user=request.user)
#         except MealPlan.DoesNotExist:
#             return Response({"error": "Meal plan not found."}, status=status.HTTP_404_NOT_FOUND)

#         try:
#             dietary_assessment = DietaryAssessment.objects.get(user=request.user)
#         except DietaryAssessment.DoesNotExist:
#             return Response({"error": "Dietary assessment not found."}, status=status.HTTP_404_NOT_FOUND)

#         tags = {
#             "health_goals": dietary_assessment.health_goals,
#             "dietary_preferences": dietary_assessment.dietary_preferences
#         }

#         meal_plan.tags = tags
#         meal_plan.save()

#         serializer = MealPlanSerializer(meal_plan)
#         return Response(serializer.data)

class RecommenderStatusView(APIView):
    def get(self, request, task_id):
        task_result = AsyncResult(task_id)
        if task_result.ready():
            recommender = cache.get('hybrid_recommender')
            if recommender:
                return Response({"status": "completed"})
            return Response({"status": "failed"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        return Response({"status": "processing"})