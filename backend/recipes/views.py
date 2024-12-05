from django.shortcuts import render
from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import generics, permissions, status, viewsets
from rest_framework.authtoken.views import ObtainAuthToken,APIView
from django.core.cache import cache
from .models import Recipe, NutritionalInformation, Rating, Favorite,TagsType,MealType,DishType,CuisineType,Ingredient
from .serializers import RecipeFilterSerializer, RecipeListSerializer, RecipeSearchSerializer,RecipeSerializer, NutritionalInformationSerializer, RatingSerializer, FavoriteSerializer,IngredientSerializer
from django.shortcuts import get_object_or_404
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser
from django.core.cache import cache
from django.utils import timezone
from django.db import transaction
import json
import logging


# Create your views here.
class RecipeListCreateView(APIView):
    """
    API view to list all recipes or create a new one.
    """
    permission_classes = [permissions.AllowAny]
    parser_classes = (MultiPartParser, FormParser,JSONParser)


    def get(self, request):
        recipes = Recipe.objects.all()
        serializer = RecipeSerializer(recipes, many=True, context={'request': request})
        return Response(serializer.data)

    def post(self, request, *args, **kwargs):
        serializer = RecipeSerializer(data=request.data, context={'request': request},many=True)
        if serializer.is_valid():
            try:
                with transaction.atomic():
                    serializer.save()
                return Response({
                    'message': 'Recipe created successfully',
                    'recipe': serializer.data
                }, status=status.HTTP_201_CREATED)
            except IntegrityError as e:
                return Response({'error': f'Error creating recipe: {str(e)}'}, status=status.HTTP_400_BAD_REQUEST)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class RecipeDetailView(APIView):
    """
    API view to retrieve, update or delete a recipe.
    """
    permission_classes = [permissions.AllowAny]
    parser_classes = (MultiPartParser, FormParser,JSONParser)

    def get_object(self, recipe_id):
        return get_object_or_404(Recipe, recipe_id=recipe_id)

    def get(self, request, recipe_id):
        recipe = self.get_object(recipe_id)
        serializer = RecipeSerializer(recipe, context={'request': request})
        return Response(serializer.data)

    def put(self, request, recipe_id):
        recipe = self.get_object(recipe_id)
        serializer = RecipeSerializer(recipe, data=request.data, partial=True, context={'request': request})
        if serializer.is_valid():
            serializer.save()
            return Response({
                'message': 'Recipe updated successfully',
                'recipe': serializer.data
            })
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request, recipe_id):
        recipe = self.get_object(recipe_id)
        recipe.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)




class RecipeFilterView(APIView):
    """
    API view to filter recipes by any column.
    """
    permission_classes = [permissions.AllowAny]

    def get(self, request):
        filters = {}
        for key, value in request.query_params.items():
            if value is not None:
                if key in ['dish_type', 'cuisine', 'tags','meal_type','ingredients']:  # Fields that are ArrayFields
                    filters[f'{key}__contains'] = [value]  # Use __contains lookup
                else:
                    filters[key] = value
        
        recipes = Recipe.objects.filter(**filters)
        serializer = RecipeSerializer(recipes, many=True, context={'request': request})
        return Response(serializer.data)

    # def get(self, request):
    #     serializer = RecipeFilterSerializer(data=request.query_params)
    #     if serializer.is_valid():
    #         filters = {}
    #         for key, value in serializer.validated_data.items():
    #             if value is not None:
    #                 if key in ['dish_type', 'cuisine', 'tags','meal_type']:  # Fields that are ArrayFields
    #                     filters[f'{key}__contains'] = [value]  # Use __contains lookup
    #                 else:
    #                     filters[key] = value
            
    #         recipes = Recipe.objects.filter(**filters)
    #         recipe_serializer = RecipeListSerializer(recipes, many=True, context={'request': request})
    #         return Response(recipe_serializer.data)
    #     return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
class RecipeSearchView(APIView):
    """
    API view to search recipes.
    """
    permission_classes = [permissions.AllowAny]

    def get(self, request):
        serializer = RecipeSearchSerializer(data=request.query_params)
        if serializer.is_valid():
            query = serializer.validated_data.get('query', '')

            # Perform the search on the Recipe model
            recipes = Recipe.objects.filter(name__icontains=query) | \
                      Recipe.objects.filter(ingredients__icontains=query) | \
                      Recipe.objects.filter(tags__icontains=query)

            # Serialize the search results
            recipe_serializer = RecipeListSerializer(recipes, many=True, context={'request': request})
            return Response(recipe_serializer.data)
        
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class NutritionInformationView(APIView):
    """
    API view to get or update nutrition information.
    """
    permission_classes = [permissions.AllowAny]

    def get_object(self, nutrition_id):
        return get_object_or_404(NutritionalInformation, nutrition_info_id=nutrition_id)

    def get(self, request, nutrition_id):
        nutrition = self.get_object(nutrition_id)
        serializer = NutritionalInformationSerializer(nutrition)
        return Response(serializer.data)

    def put(self, request, nutrition_id):
        nutrition = self.get_object(nutrition_id)
        serializer = NutritionalInformationSerializer(nutrition, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response({
                'message': 'Nutrition information updated successfully',
                'nutrition': serializer.data
            })
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class RatingView(APIView):
    """
    API view to handle rating creation and retrieval.
    """
    
    def get_permissions(self):
        # Allow GET requests to be accessible by any user
        if self.request.method == 'GET':
            return [permissions.AllowAny()]
        # For all other methods (like POST), require authentication
        return [permissions.IsAuthenticated()]

    def get(self, request, recipe_id):
        ratings = Rating.objects.filter(recipe_id=recipe_id)
        serializer = RatingSerializer(ratings, many=True)
        return Response(serializer.data)

    def post(self, request, recipe_id):
        data = request.data.copy()  # Create a mutable copy of the request data
        data['recipe'] = recipe_id  # Add recipe_id to the data

        serializer = RatingSerializer(data=data)  # Pass the modified data to the serializer
        if serializer.is_valid():
            serializer.save(user=request.user)  # Save the rating, associating with the user
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
class RecipeRatingsView(APIView):
    def get(self, request):
        recipe_ids = request.query_params.getlist('recipe_ids')  # Fetch recipe_ids from query parameters
        ratings = Rating.objects.filter(recipe_id__in=recipe_ids)
        serializer = RatingSerializer(ratings, many=True)
        return Response(serializer.data)
    
class FavoriteView(APIView):
    """
    API view to handle favorite creation and retrieval.
    """
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        favorites = Favorite.objects.filter(user=request.user)
        serializer = FavoriteSerializer(favorites, many=True)
        return Response(serializer.data)

    def post(self, request, recipe_id):
        data = {'recipe': recipe_id, 'timestamp': timezone.now()}  # Ensure recipe_id and timestamp are in the data
        serializer = FavoriteSerializer(data=data)
        if serializer.is_valid():
            serializer.save(user=request.user)  # Save the favorite and associate it with the current user
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request, recipe_id):
        try:
            favorite = Favorite.objects.get(user=request.user, recipe_id=recipe_id)
            favorite.delete()
            return Response(status=status.HTTP_204_NO_CONTENT)
        except Favorite.DoesNotExist:
            return Response({'error': 'Favorite not found'}, status=status.HTTP_404_NOT_FOUND)
        
class IngredientView(APIView):
    """
    API view to handle ingredient creation, retrieval, and update.
    """
    permission_classes = [permissions.AllowAny]  # Allow read-only access for unauthenticated users

    # GET: Retrieve all ingredients
    def get(self, request):
        ingredients = Ingredient.objects.all()
        serializer = IngredientSerializer(ingredients, many=True)
        return Response(serializer.data)

    # POST: Add a new ingredient
    def post(self, request):
        data = request.data  # Ingredient data sent in the request
        serializer = IngredientSerializer(data=data,many=True)
        if serializer.is_valid():
            serializer.save()  # Save the new ingredient to the database
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    # PUT: Update an existing ingredient
    def put(self, request, ingredients_id):
        try:
            ingredient = Ingredient.objects.get(ingredients_id=ingredients_id)
        except Ingredient.DoesNotExist:
            return Response({'error': 'Ingredient not found'}, status=status.HTTP_404_NOT_FOUND)

        serializer = IngredientSerializer(ingredient, data=request.data, partial=True)  # Support partial updates
        if serializer.is_valid():
            serializer.save()  # Update the ingredient in the database
            return Response({
                'message': 'Ingredient updated successfully',
                'ingredient': serializer.data
            }, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
class MealTypeView(APIView):
    def get(self, request):
        choices = MealType.choices
        serialized_choices = [
            {"value": meal_type[0], "display_name": meal_type[1]} for meal_type in choices
        ]
        return Response(serialized_choices, status=status.HTTP_200_OK)

class DishTypeView(APIView):
    def get(self, request):
        choices = DishType.choices
        serialized_choices = [
            {"value": dish_type[0], "display_name": dish_type[1]} for dish_type in choices
        ]
        return Response(serialized_choices, status=status.HTTP_200_OK)

class CuisineTypeView(APIView):
    def get(self, request):
        choices = CuisineType.choices
        serialized_choices = [
            {"value": cuisine[0], "display_name": cuisine[1]} for cuisine in choices
        ]
        return Response(serialized_choices, status=status.HTTP_200_OK)
class TagsTypeView(APIView):
    def get(self, request):
        choices = TagsType.choices
        serialized_choices = [
            {"value": tag[0], "display_name": tag[1]} for tag in choices
        ]
        return Response(serialized_choices, status=status.HTTP_200_OK)