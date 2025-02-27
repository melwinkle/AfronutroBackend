from rest_framework import serializers
from django.contrib.auth import get_user_model
from django.contrib.auth.password_validation import validate_password
from django.db import IntegrityError, transaction
from .models import ActivityLevel, DietaryAssessment, DietaryPreference, EducationalContent, HealthGoal
from recipes.models import Ingredient

User = get_user_model()

class UserSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True, required=True, validators=[validate_password])
    password2 = serializers.CharField(write_only=True, required=True)
    age = serializers.SerializerMethodField()
    class Meta:
        model = User
        fields = ('id', 'email','username', 'password','password2', 'age','date_of_birth','gender','height','weight','is_verified','activity_levels','tdee','bmi','last_password_change','is_active')
        extra_kwargs = { 'email': {'required': True}}
        

    def validate(self, attrs):
        if attrs['password'] != attrs['password2']:
            raise serializers.ValidationError({"password": "Password fields didn't match."})
        return attrs

    def create(self, validated_data):
        validated_data.pop('password2')
        user = User.objects.create_user(**validated_data)
        return user
    
    def get_age(self, obj):
        return obj.age

class UserProfileSerializer(serializers.ModelSerializer):
    age = serializers.SerializerMethodField()
    class Meta:
        model = User
        fields = ('id', 'email','username', 'age','date_of_birth','gender','height','weight','activity_levels','tdee','bmi','is_verified')
        read_only_fields = ('id', 'email')
        
    def get_age(self, obj):
        return obj.age

class PasswordResetRequestSerializer(serializers.Serializer):
    email = serializers.EmailField()

class PasswordResetConfirmSerializer(serializers.Serializer):
    uid = serializers.CharField()
    token = serializers.CharField()
    new_password = serializers.CharField()

class EducationalContentSerializer(serializers.ModelSerializer):
    class Meta:
        model = EducationalContent
        fields = '__all__'
        
    def get_content_image_url(self, obj):
        if obj.content_image:
            request = self.context.get('request')
            if request is not None:
                return request.build_absolute_uri(obj.content_image.url)
        return None


class DietaryAssessmentSerializer(serializers.ModelSerializer):
    liked_ingredients = serializers.SlugRelatedField(many=True, slug_field='name', queryset=Ingredient.objects.all())
    disliked_ingredients = serializers.SlugRelatedField(many=True, slug_field='name', queryset=Ingredient.objects.all())

    class Meta:
        model = DietaryAssessment
        fields = ['dietary_assessment_id', 'user', 'dietary_preferences', 'activity_levels', 'health_goals',
                  'liked_ingredients', 'disliked_ingredients','cuisine_preference' ,'tdee', 'bmi', 'assessment']
        read_only_fields = ['dietary_assessment_id','user']

    def validate_dietary_preferences(self, value):
        valid_preferences = set(choice[0] for choice in DietaryPreference.choices)
        if not set(value).issubset(valid_preferences):
            raise serializers.ValidationError("Invalid dietary preference(s) provided.")
        return value

    def validate_activity_levels(self, value):
        valid_levels = set(choice[0] for choice in ActivityLevel.choices)
        if not set(value).issubset(valid_levels):
            raise serializers.ValidationError("Invalid activity level(s) provided.")
        if not 1 <= len(value) <= 3:
            raise serializers.ValidationError("Choose between 1 and 3 activity levels.")
        return value

    def validate_health_goals(self, value):
        valid_goals = set(choice[0] for choice in HealthGoal.choices)
        if not set(value).issubset(valid_goals):
            raise serializers.ValidationError("Invalid health goal(s) provided.")
        if not 1 <= len(value) <= 3:
            raise serializers.ValidationError("Choose between 1 and 3 health goals.")
        return value
    
    def validate_cuisine_preferences(self, value):
        valid_preferences = set(choice[0] for choice in CuisineType.choices)
        if not set(value).issubset(valid_preferences):
            raise serializers.ValidationError("Invalid Cuisine choices provided.")
        return value

    def validate_liked_ingredients(self, value):
        if not isinstance(value, list):
            raise serializers.ValidationError("Liked ingredients must be a list.")
        return value

    def validate_disliked_ingredients(self, value):
        if not isinstance(value, list):
            raise serializers.ValidationError("Disliked ingredients must be a list.")
        return value


    def create(self, validated_data):
        user = self.context['request'].user
        validated_data['user'] = user
        return super().create(validated_data)

    def update(self, instance, validated_data):
        # Handle many-to-many fields separately
        liked_ingredients = validated_data.pop('liked_ingredients', None)
        disliked_ingredients = validated_data.pop('disliked_ingredients', None)

        # Update other fields
        for attr, value in validated_data.items():
            setattr(instance, attr, value)

        # Update liked ingredients
        if liked_ingredients is not None:
            # Get actual Ingredient objects based on names
            ingredient_objects = Ingredient.objects.filter(name__in=liked_ingredients)
            if len(ingredient_objects) != len(liked_ingredients):
                raise serializers.ValidationError("One or more liked ingredients do not exist.")
            instance.liked_ingredients.set(ingredient_objects)

        # Update disliked ingredients
        if disliked_ingredients is not None:
            # Get actual Ingredient objects based on names
            ingredient_objects = Ingredient.objects.filter(name__in=disliked_ingredients)
            if len(ingredient_objects) != len(disliked_ingredients):
                raise serializers.ValidationError("One or more disliked ingredients do not exist.")
            instance.disliked_ingredients.set(ingredient_objects)

        instance.save()
        return instance

# class MealPlanSerializer(serializers.ModelSerializer):
#     meals = RecipeSerializer(many=True, read_only=True)
#     nutritional_composition = serializers.SerializerMethodField()

#     class Meta:
#         model = MealPlan
#         fields = ['meal_plan_id', 'user', 'name', 'description', 'meals', 'tags', 'nutritional_composition']

#     def get_nutritional_composition(self, obj):
#         return obj.calculate_nutritional_composition()


class ChoiceSerializer(serializers.Serializer):
    value = serializers.CharField()
    display_name = serializers.CharField()