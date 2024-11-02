from rest_framework import serializers
from .models import Recipe, NutritionalInformation, Rating, Favorite, CuisineType,TagsType,MealType,DishType,Ingredient



class NutritionalInformationSerializer(serializers.ModelSerializer):
    class Meta:
        model = NutritionalInformation
        fields = '__all__'

class RecipeSerializer(serializers.ModelSerializer):
    nutrition = NutritionalInformationSerializer()

    class Meta:
        model = Recipe
        fields = ['recipe_id', 'name', 'ingredients', 'cuisine', 'recipe_info',
                  'vegan', 'vegetarian', 'gluten_free', 'pescatarian', 'halal',
                  'meal_type', 'dish_type', 'tags', 'nutrition','duration','image']

    def create(self, validated_data):
        nutrition_data = validated_data.pop('nutrition')
        nutrition = NutritionalInformation.objects.create(**nutrition_data)
        recipe = Recipe.objects.create(nutrition=nutrition, **validated_data)
        return recipe

    def update(self, instance, validated_data):
        nutrition_data = validated_data.pop('nutrition', None)
        if nutrition_data:
            nutrition_serializer = NutritionalInformationSerializer(instance.nutrition, data=nutrition_data, partial=True)
            if nutrition_serializer.is_valid():
                nutrition_serializer.save()
            else:
                raise serializers.ValidationError(nutrition_serializer.errors)
        
        for attr, value in validated_data.items():
            setattr(instance, attr, value)
        instance.save()
        return instance

class RecipeListSerializer(serializers.ModelSerializer):
    nutrition = NutritionalInformationSerializer()
    class Meta:
        model = Recipe
        fields = ['recipe_id', 'name', 'cuisine', 'meal_type', 'dish_type','nutrition','duration','image']


#not used
class RecipeFilterSerializer(serializers.Serializer):
    vegan = serializers.BooleanField(required=False)
    vegetarian = serializers.BooleanField(required=False)
    gluten_free = serializers.BooleanField(required=False)
    meal_type = serializers.ChoiceField(choices=MealType.choices, required=False)
    dish_type = serializers.ChoiceField(choices=DishType.choices, required=False)
    cuisine = serializers.ChoiceField(choices=CuisineType.choices, required=False)
    tags = serializers.ChoiceField(choices=TagsType.choices, required=False)

class RecipeSearchSerializer(serializers.Serializer):
    query = serializers.CharField(required=True, min_length=1)


class RatingSerializer(serializers.ModelSerializer):
    class Meta:
        model = Rating
        fields = ['rating_id','user', 'recipe', 'rating', 'comment', 'timestamp']
        read_only_fields = ['user','timestamp']

class FavoriteSerializer(serializers.ModelSerializer):
    class Meta:
        model = Favorite
        fields = ['favorite_id','user', 'recipe', 'added_on']
        read_only_fields = ['user','added_on']

class IngredientSerializer(serializers.ModelSerializer):
    substitutes = serializers.SlugRelatedField(many=True, slug_field='name', queryset=Ingredient.objects.all())

    class Meta:
        model = Ingredient
        fields = ['ingredients_id','name', 'calories', 'carbs', 'protein', 'fat', 'minerals', 'vitamins', 'substitutes']