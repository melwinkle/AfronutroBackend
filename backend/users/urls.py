from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from .views import DietaryAssessmentRetrieveView, DietaryAssessmentView, IngredientView, RegisterView,ResendVerificationEmailAPIView,VerifyEmailAPIView, LoginView, ProfileView,LogoutView,ChangePasswordView,PasswordResetRequestAPIView, PasswordResetConfirmAPIView,EducationalContentListCreateView,EducationalContentDetailView,EducationalContentByTypeView,RecipeListCreateView, RecipeDetailView, RecipeFilterView, RecipeSearchView, NutritionInformationView, RatingView,RecipeRatingsView, FavoriteView,RecalculateAssessmentView,TagsTypeView,ActivityLevelView,HealthGoalView,DietaryPreferenceView,MealTypeView,DishTypeView,CuisineTypeView

urlpatterns = [
    path('register/', RegisterView.as_view(), name='register'),
    path('login/', LoginView.as_view(), name='login'),
    path('profile/', ProfileView.as_view(), name='profile'),
    path('logout/', LogoutView.as_view(), name='logout'),
    path('resend-verification-email/', ResendVerificationEmailAPIView.as_view(), name='resend-verification-email'),
    path('verify-email/<uidb64>/<token>/', VerifyEmailAPIView.as_view(), name='verify-email'),
    path('change-password/', ChangePasswordView.as_view(), name='change_password'),
    path('password-reset/', PasswordResetRequestAPIView.as_view(), name='password_reset_request'),
    path('forgot-password/<uidb64>/<token>/', PasswordResetConfirmAPIView.as_view(), name='password_reset_confirm'),
    path('educational-content/', EducationalContentListCreateView.as_view(), name='educational-content-list-create'),
    path('educational-content/<str:content_id>/', EducationalContentDetailView.as_view(), name='educational-content-detail'),
    path('educational-content-filter/', EducationalContentByTypeView.as_view(), name='educational-content-by-type'),
    path('recipes/', RecipeListCreateView.as_view(), name='recipe-list-create'),
    path('recipes/<str:recipe_id>/', RecipeDetailView.as_view(), name='recipe-detail'),
    path('recipes-filter/', RecipeFilterView.as_view(), name='recipe-filter'),
    path('recipes-search/', RecipeSearchView.as_view(), name='recipe-search'),
    path('nutrition/<str:nutrition_id>/', NutritionInformationView.as_view(), name='nutrition-detail'),
    path('rating/<str:recipe_id>/', RatingView.as_view(), name='rating-view'),
    path('ratings/', RecipeRatingsView.as_view(), name='rating-views'),
    path('favorites/<str:recipe_id>/', FavoriteView.as_view(), name='favorite-view'),
    path('favorites/', FavoriteView.as_view(), name='favorite-view-all'),
    path('dietary-assessment/', DietaryAssessmentView.as_view(), name='dietary_assessment'),
    path('dietary-assessment-view/', DietaryAssessmentRetrieveView.as_view(), name='retrieve_dietary_assessment'),
    path('dietary-assessment-recalculate/', RecalculateAssessmentView.as_view(), name='recalculate_dietary_assessment'),
    path('ingredients/', IngredientView.as_view(), name='ingredients'),  # For GET (list all ingredients) and POST (add new ingredient)
    path('ingredients/<str:ingredients_id>/', IngredientView.as_view(), name='update_ingredients'),  # For PUT (update ingredient by id)
    path('tags/', TagsTypeView.as_view(), name='tags-type'),
    path('activity-level/', ActivityLevelView.as_view(), name='activity-level'),
    path('health-goal/', HealthGoalView.as_view(), name='health-goal'),
    path('dietary-preference/', DietaryPreferenceView.as_view(), name='dietary-preference'),
    path('meal-type/', MealTypeView.as_view(), name='meal-type'),
    path('dish-type/', DishTypeView.as_view(), name='dish-type'),
    path('cuisine-type/', CuisineTypeView.as_view(), name='cuisine-type'),
    # path('meal-plans/generate/', GenerateMealPlanView.as_view(), name='generate_meal_plan'),
    # path('meal-plans/<str:meal_plan_id>/save/', SaveMealPlanView.as_view(), name='save_meal_plan'),
    # path('meal-plans/<str:meal_plan_id>/customize/', CustomizeMealPlanView.as_view(), name='customize_meal_plan'),
    # path('meal-plans/<str:meal_plan_id>/nutritional-summary/', NutritionalSummaryView.as_view(), name='nutritional_summary_meal_plan'),
    # path('meal-plans/<str:meal_plan_id>/set-tags/', SetTagsView.as_view(), name='tags_meal_plan'),
    # path('meal-plans/', GenerateMealPlanView.as_view(), name='generate_meal_plan'),
    # path('meal-plans/<str:meal_plan_id>/', GenerateMealPlanView.as_view(), name='get_meal_plan_by_id'),
]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)