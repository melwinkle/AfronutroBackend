from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from .views import RecipeListCreateView, RecipeDetailView, RecipeFilterView, RecipeSearchView, NutritionInformationView, RatingView,RecipeRatingsView, FavoriteView,TagsTypeView,MealTypeView,DishTypeView,CuisineTypeView,IngredientView

urlpatterns = [
path('recipes/', RecipeListCreateView.as_view(), name='recipe-list-create'),
    path('recipes/<str:recipe_id>/', RecipeDetailView.as_view(), name='recipe-detail'),
    path('recipes-filter/', RecipeFilterView.as_view(), name='recipe-filter'),
    path('recipes-search/', RecipeSearchView.as_view(), name='recipe-search'),
    path('nutrition/<str:nutrition_id>/', NutritionInformationView.as_view(), name='nutrition-detail'),
    path('rating/<str:recipe_id>/', RatingView.as_view(), name='rating-view'),
    path('ratings/', RecipeRatingsView.as_view(), name='rating-views'),
    path('favorites/<str:recipe_id>/', FavoriteView.as_view(), name='favorite-view'),
    path('favorites/', FavoriteView.as_view(), name='favorite-view-all'),
    path('ingredients/', IngredientView.as_view(), name='ingredients'),  # For GET (list all ingredients) and POST (add new ingredient)
    path('ingredients/<str:ingredients_id>/', IngredientView.as_view(), name='update_ingredients'),  # For PUT (update ingredient by id)
    path('tags/', TagsTypeView.as_view(), name='tags-type'),
    path('meal-type/', MealTypeView.as_view(), name='meal-type'),
    path('dish-type/', DishTypeView.as_view(), name='dish-type'),
    path('cuisine-type/', CuisineTypeView.as_view(), name='cuisine-type'),
    ]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)