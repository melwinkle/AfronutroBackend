from django.contrib.auth.models import AbstractUser, BaseUserManager
from django.db import models
from django.utils.translation import gettext_lazy as _
from django.utils import timezone
from django.contrib.postgres.fields import ArrayField
from django.core.validators import MaxLengthValidator,MinLengthValidator
from recipes.models import Ingredient
import uuid
import string
import random

class UserManager(BaseUserManager):
    """Define a model manager for User model with no username field."""

    def _create_user(self, email, password=None, **extra_fields):
        """Create and save a User with the given email and password."""
        if not email:
            raise ValueError('The given email must be set')
        email = self.normalize_email(email)
        user = self.model(email=email, **extra_fields)
        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_user(self, email, password=None, **extra_fields):
        extra_fields.setdefault('is_staff', False)
        extra_fields.setdefault('is_superuser', False)
        extra_fields.setdefault('is_active', False)
        return self._create_user(email, password, **extra_fields)

    def create_superuser(self, email, password=None, **extra_fields):
        """Create and save a SuperUser with the given email and password."""
        extra_fields.setdefault('is_staff', True)
        extra_fields.setdefault('is_superuser', True)
        extra_fields.setdefault('is_superuser', True)

        if extra_fields.get('is_staff') is not True:
            raise ValueError('Superuser must have is_staff=True.')
        if extra_fields.get('is_superuser') is not True:
            raise ValueError('Superuser must have is_superuser=True.')
        if extra_fields.get('is_active') is not True:
            raise ValueError('Superuser must have is_active=True.')

        return self._create_user(email, password, **extra_fields)

class User(AbstractUser):
    username = models.CharField(max_length=30, unique=True)
    email = models.EmailField(_('email address'), unique=True)
    age = models.IntegerField(default=0)
    gender = models.CharField(max_length=20,null=True)
    height = models.FloatField(default=0.0)
    weight = models.FloatField(default=0.0)
    is_verified = models.BooleanField(default=False)
    activity_levels=models.FloatField(default=1.55)
    last_password_change = models.DateTimeField(default=timezone.now)
    tdee = models.FloatField(default=0.0)
    bmi = models.FloatField(default=0.0)

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = []

    objects = UserManager()
    
    
    class Meta:
        db_table = 'users_user'

    def __str__(self):
        return self.email
    
   

    def set_password(self, raw_password):
        self.last_password_change = timezone.now()
        super().set_password(raw_password)

class PasswordHistory(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    password = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']


def content_file_path(instance, filename):
    return f'educational_content/{instance.content_id}/{filename}'

def generate_content_id():
    # Generate a random 8-character alphanumeric string
    chars = string.ascii_uppercase + string.ascii_lowercase + string.digits
    return ''.join(random.choice(chars) for _ in range(8))

class EducationalContent(models.Model):
    content_id = models.CharField(max_length=8, primary_key=True, default=generate_content_id, editable=False)
    title = models.CharField(max_length=255)
    description = models.TextField()
    content_type = models.CharField(max_length=100)
    content_url = models.URLField(null=True)
    tags = models.JSONField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    content_image = models.FileField(upload_to=content_file_path, blank=True, null=True)
    content_main = models.TextField(default="Test")
    

    def get_content(self):
        pass

    def get_tags(self):
        return self.tags





class ActivityLevel(models.TextChoices):
    SEDENTARY = 'SED', 'Sedentary'
    LIGHTLY_ACTIVE = 'LIG', 'Lightly Active'
    MODERATELY_ACTIVE = 'MOD', 'Moderately Active'
    VERY_ACTIVE = 'VER', 'Very Active'
    EXTRA_ACTIVE = 'EXT', 'Extra Active'

class HealthGoal(models.TextChoices):
    LOSE_WEIGHT = 'LOS', 'Lose Weight'
    MAINTAIN_WEIGHT = 'MAI', 'Maintain Weight'
    GAIN_WEIGHT = 'GAI', 'Gain Weight'
    IMPROVE_FITNESS = 'FIT', 'Improve Fitness'
    INCREASE_MUSCLE = 'MUS', 'Increase Muscle'

class DietaryPreference(models.TextChoices):
    GLUTEN_FREE = 'GLU', 'Gluten-Free'
    LACTOSE_FREE = 'LAC', 'Lactose-Free'
    NUT_FREE = 'NUT', 'Nut-Free'
    SHELLFISH_FREE = 'SHE', 'Shellfish-Free'
    EGG_FREE = 'EGG', 'Egg-Free'
    SOY_FREE = 'SOY', 'Soy-Free'
    PEANUT_FREE = 'PEA', 'Peanut-Free'
    KOSHER = 'KOS', 'Kosher'
    HALAL = 'HAL', 'Halal'
    VEGAN = 'VEG', 'Vegan'
    VEGETARIAN = 'VGT', 'Vegetarian'
    LOW_SUGAR = 'LSU', 'Low Sugar'
    DIABETIC = 'DIA', 'Diabetic'
    SPICY_FOOD = 'SPI', 'Spicy Food'
    SWEET_FOOD = 'SWE', 'Sweet Food'
    SAVORY_FOOD = 'SAV', 'Savory Food'
    ORGANIC = 'ORG', 'Organic'
    HIGH_PROTEIN = 'HPR', 'High Protein'
    LOW_CARB = 'LCA', 'Low Carb'
    HIGH_FIBER = 'HFI', 'High Fiber'
    KETO = 'KET', 'Keto'
    PALEO = 'PAL', 'Paleo'
    DAIRY_FREE = 'DAI', 'Dairy-Free'


    
class DietaryAssessment(models.Model):
    dietary_assessment_id = models.CharField(max_length=8,default=generate_content_id, primary_key=True, editable=False)
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    dietary_preferences = models.JSONField()  # Storing list as JSON
    activity_levels = models.JSONField(validators=[MinLengthValidator(1), MaxLengthValidator(3)])  # Storing up to 3 choices as JSON
    health_goals = models.JSONField(validators=[MinLengthValidator(1), MaxLengthValidator(3)])  # Storing up to 3 choices as JSON
    liked_ingredients = models.ManyToManyField(Ingredient, related_name='liked_by_assessments')
    disliked_ingredients = models.ManyToManyField(Ingredient, related_name='disliked_by_assessments')
    cuisine_preference=models.JSONField(default=dict)
    tdee = models.FloatField(default=0.0)
    bmi = models.FloatField(default=0.0)
    assessment = models.TextField()

