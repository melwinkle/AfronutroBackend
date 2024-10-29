from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from .views import GenerateMealPlanView,SaveMealPlanView,CustomizeMealPlanView,NutritionalSummaryView

urlpatterns = [
    path('meal-plans/generate/', GenerateMealPlanView.as_view(), name='generate_meal_plan'),
    path('meal-plans/<str:meal_plan_id>/save/', SaveMealPlanView.as_view(), name='save_meal_plan'),
    path('meal-plans/<str:meal_plan_id>/customize/', CustomizeMealPlanView.as_view(), name='customize_meal_plan'),
    path('meal-plans/<str:meal_plan_id>/nutritional-summary/', NutritionalSummaryView.as_view(), name='nutritional_summary_meal_plan'),
    path('meal-plans/', GenerateMealPlanView.as_view(), name='generate_meal_plan'),
    path('meal-plans/<str:meal_plan_id>/', GenerateMealPlanView.as_view(), name='get_meal_plan_by_id'),
]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)