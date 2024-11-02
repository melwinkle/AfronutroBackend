from rest_framework import serializers
from .models import MealPlan
from recipes.serializers import RecipeSerializer


class MealPlanSerializer(serializers.ModelSerializer):
    meals = RecipeSerializer(many=True, read_only=True)
    nutritional_composition = serializers.SerializerMethodField()

    class Meta:
        model = MealPlan
        fields = ['meal_plan_id', 'user', 'name', 'description', 'meals','meals_structure', 'tags', 'nutritional_composition','status']

    def get_nutritional_composition(self, obj):
        return obj.calculate_nutritional_composition()
    
    def to_representation(self, instance):
        data = super().to_representation(instance)
        # Ensure meals_structure is properly serialized
        if isinstance(instance.meals_structure, str):
            data['meals_structure'] = json.loads(instance.meals_structure)
        return data